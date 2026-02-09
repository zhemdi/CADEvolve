import re, os, ast, argparse, resource, traceback
import numpy as np
from pathlib import Path
from utils.canonicalization.helpers import ProcessPool, log_line, save_code

SELECTOR_WP_METHODS = {"solids", "faces", "edges", "wires", "shells", "vertices"}

SELECTOR_CANDIDATES = [
    ">Z", "<Z", "|Z",
    ">X", "<X", "|X",
    ">Y", "<Y", "|Y",
]

class Binarize(ast.NodeTransformer):

    def __init__(self, zero_tol: float = 1e-7, min_positive: int = 1):
        self.zero_tol = zero_tol
        self.min_positive = int(min_positive)

        self.nonzero_methods = {
            "fillet":  {"pos": [0], "mode": "gt0"},
            "chamfer": {"pos": [0], "mode": "gt0"},
            "circle": {"pos": [0], "mode": "gt0"},
            "ellipse": {"pos": [0, 1], "mode": "gt0"},
            "sphere": {"pos": [0], "mode": "gt0"},
            "hole": {"pos": [0], "mode": "gt0"},
            "cborehole": {"pos": [0, 1, 2], "mode": "gt0"},
            "cskhole": {"pos": [0, 1, 2], "mode": "gt0"},
            "extrude": {"pos": [0], "mode": "ne0"},
            "shell": {"pos": [0], "mode": "gt0"},
            "rect": {"pos": [0, 1], "mode": "gt0"},
            "splineapprox": {"pos": [1, 2, 3], "kw": ["tol", "minDeg", "maxDeg", "mindeg", "maxdeg"], "mode": "gt0"},
        }

        self._point_seq_methods = {"polyline", "splineapprox", "spline", "polygon"}

        self._point_kw_names = {"pts", "points", "pnts"}

    def clean(self, value):
        if abs(value) < self.zero_tol:
            return 0
        return int(round(value))

    def visit_Constant(self, node):
        v = node.value

        if isinstance(v, bool):
            return node

        if isinstance(v, (int, float)):
            new_v = self.clean(v)
            return ast.copy_location(ast.Constant(value=new_v), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = self.generic_visit(node)
        if isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
            v = node.operand.value

            if isinstance(v, bool):
                return node

            if isinstance(v, (int, float)):
                x = +v if isinstance(node.op, ast.UAdd) else -v
                new_v = self.clean(x)
                return ast.copy_location(ast.Constant(value=new_v), node)
        return node

    def _needs_clamp(self, v: int, mode: str) -> bool:
        if mode == "ne0":
            return v == 0
        return v <= 0

    def _clamp_const(self, const_node: ast.Constant, mode: str) -> ast.Constant:
        v = const_node.value

        if isinstance(v, bool):
            return const_node

        if isinstance(v, (int, float)):
            iv = int(v)
            if self._needs_clamp(iv, mode):
                return ast.copy_location(ast.Constant(value=self.min_positive), const_node)
        return const_node

    def _point_key(self, node: ast.AST):
        if not isinstance(node, (ast.List, ast.Tuple)):
            return None

        coords = []
        for e in node.elts:
            if isinstance(e, ast.Constant):
                v = e.value
                if isinstance(v, bool):
                    return None
                if isinstance(v, (int, float)):
                    coords.append(int(v))
                else:
                    return None
            else:
                return None

        if not coords:
            return None
        return tuple(coords)

    def _dedupe_consecutive_points(self, seq_node: ast.AST) -> ast.AST:
        if not isinstance(seq_node, (ast.List, ast.Tuple)):
            return seq_node

        new_elts = []
        prev_key = object()

        for elt in seq_node.elts:
            k = self._point_key(elt)
            if k is not None and k == prev_key:
                continue
            new_elts.append(elt)
            prev_key = k

        if len(new_elts) == len(seq_node.elts):
            return seq_node

        if isinstance(seq_node, ast.List):
            out = ast.List(elts=new_elts, ctx=seq_node.ctx)
        else:
            out = ast.Tuple(elts=new_elts, ctx=seq_node.ctx)

        return ast.copy_location(out, seq_node)


    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            mlow = method.lower()

            spec = self.nonzero_methods.get(mlow)
            if spec:
                mode = spec.get("mode", "gt0")

                for idx in spec.get("pos", []):
                    if 0 <= idx < len(node.args) and isinstance(node.args[idx], ast.Constant):
                        node.args[idx] = self._clamp_const(node.args[idx], mode)

                kw_names = set(spec.get("kw", []))
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    if kw.arg in kw_names and isinstance(kw.value, ast.Constant):
                        kw.value = self._clamp_const(kw.value, mode)

            if mlow in self._point_seq_methods:
                if len(node.args) >= 1:
                    node.args[0] = self._dedupe_consecutive_points(node.args[0])

                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    if kw.arg.lower() in self._point_kw_names:
                        kw.value = self._dedupe_consecutive_points(kw.value)

        return node
    

def process_binarization_task(task, max_attempts: int = 5):
    dir_name     = task.get("dir")
    file_name    = task["file"]
    script_path  = Path(task["path"])
    dest_root    = Path(task["dest_root"])
    log_path     = task["log_path"]

    is_flat = (dir_name is None)
    detail_key = (Path(file_name).stem if is_flat else dir_name)
    script_key = (None if is_flat else Path(file_name).stem)

    def exec_capture_lineno(code: str):
        """
        Execute `code` and return:
          (ok: bool, exc: Exception|None, lineno: int|None, tb_str: str|None)
        lineno is extracted from the deepest traceback frame with filename == "<string>".
        """
        ns = {"np": np}
        try:
            exec(code, ns, ns)
            return True, None, None, None
        except Exception as e:
            tb = traceback.TracebackException.from_exception(e)
            lineno = None
            for frame in reversed(tb.stack):
                if frame.filename == "<string>":
                    lineno = frame.lineno
                    break
            return False, e, lineno, "".join(tb.format())

    def call_method_name(node: ast.Call) -> str:
        f = node.func
        if isinstance(f, ast.Attribute):
            return f.attr
        if isinstance(f, ast.Name):
            return f.id
        return ""

    def find_innermost_call_covering_line(tree: ast.AST, lineno: int) -> ast.Call | None:
        """
        Find the innermost Call such that call.lineno <= lineno <= call.end_lineno.
        Tie-breaker: minimal span, then maximal col_offset.
        """
        best = None
        best_span = None
        best_col = None

        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            if not (hasattr(n, "lineno") and hasattr(n, "end_lineno")):
                continue
            if n.lineno <= lineno <= n.end_lineno:
                span = n.end_lineno - n.lineno
                col = getattr(n, "col_offset", 0)
                if best is None:
                    best, best_span, best_col = n, span, col
                else:
                    if span < best_span or (span == best_span and col >= best_col):
                        best, best_span, best_col = n, span, col
        return best

    def _is_edges_selected_error(exc: Exception) -> bool:
        # "Fillets requires that edges be selected"
        # "Chamfers requires that edges be selected"
        return "requires that edges be selected" in str(exc)

    def _get_selector_string_from_call(sel_call: ast.Call) -> str | None:
        # edges('|Z') -> args[0]
        if sel_call.args and isinstance(sel_call.args[0], ast.Constant) and isinstance(sel_call.args[0].value, str):
            return sel_call.args[0].value

        for kw in sel_call.keywords:
            if kw.arg is None:
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value

        return None

    def _find_last_selector_call_in_expr(expr: ast.AST) -> ast.Call | None:
        """
        Find the latest selector call in expr:
        <something>.<method>('<selector>')
        where method in SELECTOR_WP_METHODS.
        (requires an explicit string selector)
        """
        best = None
        best_ln = -1
        best_col = -1

        for n in ast.walk(expr):
            if not isinstance(n, ast.Call):
                continue
            if not isinstance(n.func, ast.Attribute):
                continue

            m = n.func.attr.lower()
            if m not in SELECTOR_WP_METHODS:
                continue

            if _get_selector_string_from_call(n) is None:
                continue

            ln = getattr(n, "lineno", -1)
            col = getattr(n, "col_offset", -1)
            if ln > best_ln or (ln == best_ln and col > best_col):
                best = n
                best_ln = ln
                best_col = col

        return best

    def _find_last_selector_call_any_in_expr(expr: ast.AST) -> ast.Call | None:
        """
        Find the latest selector call in expr:
        <something>.<method>(...)
        where method in SELECTOR_WP_METHODS.
        (does NOT require a selector string; matches .edges() too)
        """
        best = None
        best_ln = -1
        best_col = -1

        for n in ast.walk(expr):
            if not isinstance(n, ast.Call):
                continue
            if not isinstance(n.func, ast.Attribute):
                continue

            m = n.func.attr.lower()
            if m not in SELECTOR_WP_METHODS:
                continue

            ln = getattr(n, "lineno", -1)
            col = getattr(n, "col_offset", -1)
            if ln > best_ln or (ln == best_ln and col > best_col):
                best = n
                best_ln = ln
                best_col = col

        return best

    def _build_assign_map_upto(tree: ast.AST, upto_lineno: int):
        """
        Map var_name -> list[(lineno, rhs_expr)] for assignments strictly before upto_lineno.
        """
        m = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and hasattr(n, "lineno") and n.lineno < upto_lineno:
                if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    name = n.targets[0].id
                    m.setdefault(name, []).append((n.lineno, n.value))

            elif isinstance(n, ast.AnnAssign) and hasattr(n, "lineno") and n.lineno < upto_lineno:
                if isinstance(n.target, ast.Name) and n.value is not None:
                    name = n.target.id
                    m.setdefault(name, []).append((n.lineno, n.value))

        for k in m:
            m[k].sort(key=lambda x: x[0])
        return m

    def _resolve_name_rhs(assign_map, name: str, upto_lineno: int):
        """
        RHS expr of the latest assignment to `name` before upto_lineno.
        """
        lst = assign_map.get(name)
        if not lst:
            return None

        rhs = None
        for ln, v in lst:
            if ln < upto_lineno:
                rhs = v
            else:
                break
        return rhs

    def _patch_selector_call_in_tree(tree: ast.AST, target_lineno: int, target_col: int, target_method: str, new_selector: str) -> bool:
        """
        Find selector Call by (lineno, col_offset, method) and set/insert selector string.
        Supports:
          - edges('|Z')  -> replace
          - edges(selector='|Z') -> replace
          - edges() -> INSERT edges('<cand>')
        """
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            if getattr(n, "lineno", None) != target_lineno:
                continue
            if getattr(n, "col_offset", None) != target_col:
                continue
            if not isinstance(n.func, ast.Attribute):
                continue
            if n.func.attr.lower() != target_method:
                continue

            # positional replace
            if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                n.args[0] = ast.copy_location(ast.Constant(value=new_selector), n.args[0])
                return True

            # keyword replace (any string kw)
            for kw in n.keywords:
                if kw.arg is None:
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    kw.value = ast.copy_location(ast.Constant(value=new_selector), kw.value)
                    return True

            # INSERT for edges() / faces() / etc with no selector
            if len(n.args) == 0:
                n.args = [ast.copy_location(ast.Constant(value=new_selector), n)]
                return True

            # otherwise (has non-string args) — do not touch
            return False

        return False

    def _try_fix_selector_for_edges_selected_error(code: str, failing_lineno: int) -> str | None:
        """
        If failing call is fillet/chamfer and receiver is based on a selector from SELECTOR_WP_METHODS,
        try replacing that selector with candidates until the whole code executes.
        Return fixed code on success, else None.
        """
        tree0 = ast.parse(code)

        bad_call = find_innermost_call_covering_line(tree0, failing_lineno)
        if bad_call is None or not isinstance(bad_call.func, ast.Attribute):
            return None

        op = bad_call.func.attr.lower()
        if op not in {"fillet", "chamfer"}:
            return None

        recv = bad_call.func.value  # e.g. wp11 in wp11.fillet(...)

        sel_call = _find_last_selector_call_in_expr(recv)

        if sel_call is None and isinstance(recv, ast.Name):
            assign_map = _build_assign_map_upto(tree0, failing_lineno)
            rhs = _resolve_name_rhs(assign_map, recv.id, failing_lineno)
            if rhs is not None:
                sel_call = _find_last_selector_call_in_expr(rhs)

        if sel_call is None or not isinstance(sel_call.func, ast.Attribute):
            return None

        sel_method = sel_call.func.attr.lower()
        if sel_method not in SELECTOR_WP_METHODS:
            return None

        sel_ln = getattr(sel_call, "lineno", None)
        sel_col = getattr(sel_call, "col_offset", None)
        if sel_ln is None or sel_col is None:
            return None

        current_sel = _get_selector_string_from_call(sel_call)

        for cand in SELECTOR_CANDIDATES:
            if current_sel is not None and cand == current_sel:
                continue

            t = ast.parse(code)
            if not _patch_selector_call_in_tree(t, sel_ln, sel_col, sel_method, cand):
                continue

            ast.fix_missing_locations(t)
            cand_code = ast.unparse(t)

            ok, _, _, _ = exec_capture_lineno(cand_code)
            if ok:
                return cand_code

        return None

    def _try_rotate_selector_for_fillet_or_chamfer_failure(code: str, failing_lineno: int) -> str | None:
        """
        When failing on fillet/chamfer (any error), try rotating selector of the last
        edges/faces/... call used to build the receiver chain.
        Works even if selector is missing (edges()) -> will insert candidate.
        """
        tree0 = ast.parse(code)

        bad_call = find_innermost_call_covering_line(tree0, failing_lineno)
        if bad_call is None or not isinstance(bad_call.func, ast.Attribute):
            return None

        op = bad_call.func.attr.lower()
        if op not in {"fillet", "chamfer"}:
            return None

        recv = bad_call.func.value
        sel_call = _find_last_selector_call_any_in_expr(recv)

        if sel_call is None and isinstance(recv, ast.Name):
            assign_map = _build_assign_map_upto(tree0, failing_lineno)
            rhs = _resolve_name_rhs(assign_map, recv.id, failing_lineno)
            if rhs is not None:
                sel_call = _find_last_selector_call_any_in_expr(rhs)

        if sel_call is None or not isinstance(sel_call.func, ast.Attribute):
            return None

        sel_method = sel_call.func.attr.lower()
        if sel_method not in SELECTOR_WP_METHODS:
            return None

        sel_ln = getattr(sel_call, "lineno", None)
        sel_col = getattr(sel_call, "col_offset", None)
        if sel_ln is None or sel_col is None:
            return None

        current_sel = _get_selector_string_from_call(sel_call)

        for cand in SELECTOR_CANDIDATES:
            if current_sel is not None and cand == current_sel:
                continue

            t = ast.parse(code)
            if not _patch_selector_call_in_tree(t, sel_ln, sel_col, sel_method, cand):
                continue

            ast.fix_missing_locations(t)
            cand_code = ast.unparse(t)

            ok, _, _, _ = exec_capture_lineno(cand_code)
            if ok:
                return cand_code

        return None

    def bump_int(v: int, direction: int) -> int:
        """
        Apply +/-1 in a sign-aware way.
        - 0 -> +1 or -1
        - >0 -> v + direction
        - <0 -> v - direction (keeps sign direction consistent)
        """
        if v == 0:
            return 1 if direction > 0 else -1
        if v > 0:
            return v + direction
        return v - direction

    def adjust_expr_numbers(expr: ast.AST, direction: int) -> ast.AST:
        class _ExprAdjust(ast.NodeTransformer):
            def visit_Constant(self, node: ast.Constant):
                v = node.value

                if isinstance(v, bool):
                    return node

                if isinstance(v, (int, float)):
                    iv = int(v)
                    return ast.copy_location(ast.Constant(value=bump_int(iv, direction)), node)

                return node

            def visit_UnaryOp(self, node: ast.UnaryOp):
                node = self.generic_visit(node)

                if isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
                    v = node.operand.value

                    if isinstance(v, bool):
                        return node

                    if isinstance(v, (int, float)):
                        base = int(v)
                        signed = +base if isinstance(node.op, ast.UAdd) else -base
                        new_signed = bump_int(signed, direction)
                        return ast.copy_location(ast.Constant(value=new_signed), node)

                return node

        return _ExprAdjust().visit(expr)

    ADJUST_SPECS = {
        "splineapprox": {"pos": [1, 2, 3], "kw": {"tol", "minDeg", "maxDeg", "mindeg", "maxdeg"}},
    }

    class AdjustOnlyTargetCall(ast.NodeTransformer):
        """
        Adjust numbers ONLY inside the selected Call's relevant args/keywords.
        For splineApprox: only tol/minDeg/maxDeg.
        For others: adjust all args + all keywords.
        """
        def __init__(self, target: ast.Call, direction: int):
            self.direction = int(direction)

            self.t_l  = getattr(target, "lineno", None)
            self.t_c  = getattr(target, "col_offset", None)
            self.t_el = getattr(target, "end_lineno", None)
            self.t_ec = getattr(target, "end_col_offset", None)

            self.method = call_method_name(target).lower()
            self.spec = ADJUST_SPECS.get(self.method, None)

        def _is_target(self, node: ast.Call) -> bool:
            return (
                getattr(node, "lineno", None) == self.t_l and
                getattr(node, "col_offset", None) == self.t_c and
                getattr(node, "end_lineno", None) == self.t_el and
                getattr(node, "end_col_offset", None) == self.t_ec
            )

        def visit_Call(self, node: ast.Call):
            node = self.generic_visit(node)

            if not self._is_target(node):
                return node

            if self.spec is None:
                node.args = [adjust_expr_numbers(a, self.direction) for a in node.args]
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    kw.value = adjust_expr_numbers(kw.value, self.direction)
                return node

            pos_idx = self.spec["pos"]
            kw_set  = self.spec["kw"]

            for i in pos_idx:
                if 0 <= i < len(node.args):
                    node.args[i] = adjust_expr_numbers(node.args[i], self.direction)

            for kw in node.keywords:
                if kw.arg is None:
                    continue
                if kw.arg in kw_set:
                    kw.value = adjust_expr_numbers(kw.value, self.direction)

            return node

    # ---------------- MAIN ----------------

    try:
        orig_code = script_path.read_text(encoding="utf-8")

        ok, exc, lineno, tb_str = exec_capture_lineno(orig_code)
        if not ok:
            log_line(
                log_path,
                detail_key,
                f"{file_name}: ORIGINAL_FAILED: {type(exc).__name__}: {exc}",
                script=script_key,
            )
            return

        tree = ast.parse(orig_code)
        tree = Binarize().visit(tree)
        ast.fix_missing_locations(tree)
        new_code = ast.unparse(tree)

        last_exc = None
        last_lineno = None

        for attempt in range(1, max_attempts + 1):
            ok, exc, lineno, tb_str = exec_capture_lineno(new_code)
            if ok:
                if is_flat:
                    dest_dir = dest_root
                else:
                    dest_dir = dest_root / dir_name
                    dest_dir.mkdir(parents=True, exist_ok=True)

                dest_path = dest_dir / (Path(file_name).stem + "_binarized.py")
                save_code(str(dest_path), new_code)

                log_line(
                    log_path,
                    detail_key,
                    f"{file_name}: FINISHED: {dest_path.name}",
                    script=script_key,
                )
                return

            last_exc = exc
            last_lineno = lineno

            if lineno is None:
                break

            if _is_edges_selected_error(exc):
                fixed = _try_fix_selector_for_edges_selected_error(new_code, lineno)
                if fixed is not None:
                    new_code = fixed
                    continue
                break

            tree_mod = ast.parse(new_code)
            target = find_innermost_call_covering_line(tree_mod, lineno)
            if target is None:
                break

            direction = +1 if (attempt % 2 == 1) else -1
            tree_mod = AdjustOnlyTargetCall(target, direction).visit(tree_mod)
            ast.fix_missing_locations(tree_mod)
            new_code = ast.unparse(tree_mod)

        # try rotating selector candidates on the selector call used before fillet/chamfer.
        if last_lineno is not None:
            fixed2 = _try_rotate_selector_for_fillet_or_chamfer_failure(new_code, last_lineno)
            if fixed2 is not None:
                ok2, _, _, _ = exec_capture_lineno(fixed2)
                if ok2:
                    if is_flat:
                        dest_dir = dest_root
                    else:
                        dest_dir = dest_root / dir_name
                        dest_dir.mkdir(parents=True, exist_ok=True)

                    dest_path = dest_dir / (Path(file_name).stem + "_binarized.py")
                    save_code(str(dest_path), fixed2)

                    log_line(
                        log_path,
                        detail_key,
                        f"{file_name}: FINISHED_AFTER_SELECTOR_ROTATION: {dest_path.name}",
                        script=script_key,
                    )
                    return

        if last_exc is None:
            log_line(
                log_path,
                detail_key,
                f"{file_name}: BINARIZED_FAILED: unknown",
                script=script_key,
            )
        else:
            where = f" line={last_lineno}" if last_lineno is not None else ""
            log_line(
                log_path,
                detail_key,
                f"{file_name}: BINARIZED_FAILED:{where}: {type(last_exc).__name__}: {last_exc}",
                script=script_key,
            )
        return

    except Exception as e:
        log_line(
            log_path,
            detail_key,
            f"{file_name}: ERROR: {type(e).__name__}: {e}",
            script=script_key,
        )
        return


def run_binarization(
    scaled_dir: Path,
    binarized_dir: Path,
    logs_path: Path,
    n_workers: int,
    task_timeout: int,
    flat: bool = False,
):
    tasks = []

    scaled_dir = str(scaled_dir)
    binarized_dir = str(binarized_dir)
    logs_path = str(logs_path)

    if flat:
        already = set()
        if os.path.isdir(binarized_dir):
            for f in os.scandir(binarized_dir):
                if f.is_file() and f.name.endswith("_binarized.py"):
                    already.add(f.name)

        for f in os.scandir(scaled_dir):
            if not f.is_file():
                continue
            if not f.name.endswith(".py"):
                continue

            stem = f.name[:-3]
            final_name = f"{stem}_binarized.py"
            if final_name in already:
                continue

            tasks.append({
                "file": f.name,
                "path": f.path,
                "dest_root": binarized_dir,
                "log_path": logs_path,
            })

    else:
        processed_map = {}
        if os.path.isdir(binarized_dir):
            for entry in os.scandir(binarized_dir):
                if not entry.is_dir():
                    continue

                dir_name = entry.name
                stems = set()
                for f in os.scandir(entry.path):
                    if f.is_file() and f.name.endswith("_binarized.py"):
                        stems.add(f.name[:-len("_binarized.py")])
                processed_map[dir_name] = stems

        for dir_entry in os.scandir(scaled_dir):
            if not dir_entry.is_dir():
                continue

            dir_name = dir_entry.name
            existing_binarized = processed_map.get(dir_name, set())

            for file_entry in os.scandir(dir_entry.path):
                if not file_entry.is_file():
                    continue
                name = file_entry.name
                if not name.endswith(".py"):
                    continue

                stem = name[:-3]
                if stem in existing_binarized:
                    continue

                tasks.append({
                    "dir": dir_name,
                    "file": name,
                    "path": file_entry.path,
                    "dest_root": binarized_dir,
                    "log_path": logs_path,
                })

    pool = ProcessPool(
        task_func=process_binarization_task,
        task_args=tasks,
        n_processes=n_workers,
        timeout=task_timeout
    )

    unprocessed_args = pool.run()

    log_line(
        str(logs_path),
        "SYSTEM",
        f"binarization pipeline done (tasks_total={len(tasks)}, unprocessed={len(unprocessed_args)})",
    )

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument('--scaled-dir', type=Path, required=True)
    p.add_argument('--binarized-dir', type=Path, required=True)
    p.add_argument('--logs-path', type=Path, required=True)
    p.add_argument('--system-reserved-ratio', type=float, default=0.2)
    p.add_argument('--n-workers', type=int, default=16)
    p.add_argument('--task-timeout', type=int, default=60)
    p.add_argument('--flat', action='store_true')
    a = p.parse_args()

    scaled_dir, binarized_dir, logs_path, system_reserved_ratio, n_workers, task_timeout = (
        a.scaled_dir, a.binarized_dir, a.logs_path, a.system_reserved_ratio, a.n_workers, a.task_timeout
    )

    binarized_dir.mkdir(exist_ok=True, parents=True)
    logs_path.parent.mkdir(exist_ok=True, parents=True)

    run_binarization(
        scaled_dir=scaled_dir,
        binarized_dir=binarized_dir,
        logs_path=logs_path,
        n_workers=n_workers,
        task_timeout=task_timeout,
        flat=a.flat,
    )
