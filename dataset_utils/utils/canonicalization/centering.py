import os, ast, argparse, trimesh
import numpy as np
from pathlib import Path
from utils.canonicalization.helpers import ProcessPool, find_center, tessellate_solid, log_line, save_code
from scipy.spatial import cKDTree


def as_shape(obj):
    if hasattr(obj, "val") and callable(obj.val):
        return obj.val()
    return obj


def bbox_of(obj):
    return as_shape(obj).BoundingBox()


def verify_centering(original_code: str, new_code: str) -> bool:

    ns_orig = {"np": np}
    exec(original_code, ns_orig, ns_orig)
    result_orig = ns_orig.get('result')

    ns_new = {"np": np}
    exec(new_code, ns_new, ns_new)
    result_new = ns_new.get('result')

    c_orig = find_center(bbox_of(result_orig))
    c_new  = find_center(bbox_of(result_new))

    translation_vector = (c_orig[0] - c_new[0], c_orig[1] - c_new[1], c_orig[2] - c_new[2])

    shape_new  = as_shape(result_new)
    shape_orig = as_shape(result_orig)
    result_rolled_back = shape_new.translate(translation_vector)

    is_different = True

    vertices_orig, triangles_orig = tessellate_solid(shape_orig.wrapped)
    mesh_orig = trimesh.Trimesh(vertices=vertices_orig, faces=triangles_orig)

    vertices_rolled_back, triangles_rolled_back = tessellate_solid(result_rolled_back.wrapped)
    mesh_rolled_back = trimesh.Trimesh(vertices=vertices_rolled_back, faces=triangles_rolled_back)

    extent_orig = mesh_orig.extents.max()
    extent_rolled_back = mesh_rolled_back.extents.max()

    mesh_rolled_back.apply_transform(trimesh.transformations.scale_matrix(1 / extent_rolled_back))
    mesh_rolled_back.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

    mesh_orig.apply_transform(trimesh.transformations.scale_matrix(1 / extent_orig))
    mesh_orig.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

    n_points = 100_000
    gt_points, _ = trimesh.sample.sample_surface(mesh_orig, n_points)
    pred_points, _ = trimesh.sample.sample_surface(mesh_rolled_back, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))

    return cd > 0.15
    

class CenteringWorkplane(ast.NodeTransformer):

    def __init__(self, c):
        self.c = c

    @staticmethod
    def _num(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) \
           and isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, (int, float)):
            return -float(n.operand.value)
        raise TypeError("non-constant numeric expr")

    @staticmethod
    def sub(expr: ast.AST, delta: float) -> ast.AST:
        val = CenteringWorkplane._num(expr) - float(delta)
        return ast.copy_location(ast.Constant(value=val), expr)

    def visit_Assign(self, node):
        self.generic_visit(node)
        # wp = cq.Workplane(...)
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "cq"
            and node.value.func.attr == "Workplane"
            and node.value.keywords == []
            and (
                not node.value.args
                or (
                    len(node.value.args) == 1
                    and isinstance(node.value.args[0], ast.Constant)
                    and isinstance(node.value.args[0].value, str)
                )
            )
        ):
            new_node = ast.Assign(
                targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="cq", ctx=ast.Load()),
                        attr="Workplane",
                        ctx=ast.Load()
                    ),
                    args=(node.value.args if node.value.args else [ast.Constant(value="XY")]),
                    keywords=[
                        ast.keyword(
                            arg='origin',
                            value=ast.Tuple(
                                elts=[
                                    ast.Constant(value=-self.c[0]),
                                    ast.Constant(value=-self.c[1]),
                                    ast.Constant(value=-self.c[2])],
                                ctx=ast.Load()
                            )
                        )
                    ]
                )
            )
            return ast.copy_location(new_node, node)

        # wp = cq.Workplane(..., origin=(x, y, z), ...)
        elif (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "cq"
            and node.value.func.attr == "Workplane"
            and node.value.keywords != []
        ):
            origin_kw = next((kw for kw in node.value.keywords if kw.arg == "origin"), None)
            if origin_kw and isinstance(origin_kw.value, ast.Tuple) and len(origin_kw.value.elts) == 3:
                x, y, z = origin_kw.value.elts
                origin_kw.value = ast.Tuple(
                    elts=[self.sub(x, self.c[0]), self.sub(y, self.c[1]), self.sub(z, self.c[2])],
                    ctx=ast.Load()
                )
                return node

        # wp = cq.Workplane(cq.Plane(origin=cq.Vector(...), ...))
        elif (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "cq"
            and node.value.func.attr == "Workplane"
            and node.value.args
            and isinstance(node.value.args[0], ast.Call)
            and isinstance(node.value.args[0].func, ast.Attribute)
            and isinstance(node.value.args[0].func.value, ast.Name)
            and node.value.args[0].func.value.id == "cq"
            and node.value.args[0].func.attr == "Plane"
        ):
            plane_call = node.value.args[0]
            origin_kw = next((kw for kw in plane_call.keywords if kw.arg == "origin"), None)
            if (origin_kw and isinstance(origin_kw.value, ast.Call)
                and isinstance(origin_kw.value.func, ast.Attribute)
                and isinstance(origin_kw.value.func.value, ast.Name)
                and origin_kw.value.func.value.id == "cq"
                and origin_kw.value.func.attr == "Vector"
                and len(origin_kw.value.args) == 3):
                vx, vy, vz = origin_kw.value.args
                new_vec = ast.Call(
                    func=origin_kw.value.func,
                    args=[self.sub(vx, self.c[0]), self.sub(vy, self.c[1]), self.sub(vz, self.c[2])],
                    keywords=[]
                )
                origin_kw.value = ast.copy_location(new_vec, origin_kw.value)
                return node
            return node
        
        # wp = wp.rotate(...)
        elif (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "rotate"
        ):
            def _safe_sub(expr: ast.AST, delta: float) -> ast.AST:
                try:
                    return self.sub(expr, delta)
                except Exception:
                    return expr

            def _shift_axis_arg(arg: ast.AST) -> ast.AST:
                if isinstance(arg, (ast.List, ast.Tuple)) and len(arg.elts) == 3:
                    new_elts = [
                        _safe_sub(arg.elts[0], self.c[0]),
                        _safe_sub(arg.elts[1], self.c[1]),
                        _safe_sub(arg.elts[2], self.c[2]),
                    ]
                    new_arg = type(arg)(elts=new_elts, ctx=ast.Load())
                    return ast.copy_location(new_arg, arg)
                if (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Attribute)
                    and isinstance(arg.func.value, ast.Name)
                    and arg.func.value.id == "cq"
                    and arg.func.attr == "Vector"
                    and len(arg.args) == 3
                ):
                    new_args = [
                        _safe_sub(arg.args[0], self.c[0]),
                        _safe_sub(arg.args[1], self.c[1]),
                        _safe_sub(arg.args[2], self.c[2]),
                    ]
                    new_call = ast.Call(func=arg.func, args=new_args, keywords=[])
                    return ast.copy_location(new_call, arg)
                return arg

            if len(node.value.args) >= 2:
                node.value.args[0] = _shift_axis_arg(node.value.args[0])
                node.value.args[1] = _shift_axis_arg(node.value.args[1])

            for kw in (node.value.keywords or []):
                if kw.arg in ("axisStart", "axisStartPoint", "start", "p1"):
                    kw.value = _shift_axis_arg(kw.value)
                elif kw.arg in ("axisEnd", "axisEndPoint", "end", "p2"):
                    kw.value = _shift_axis_arg(kw.value)

            return node

        # wp = wp.revolve(...)
        elif (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "revolve"
        ):
            call = node.value
        
            def has_kw(name: str) -> bool:
                return any(isinstance(kw, ast.keyword) and kw.arg == name for kw in (call.keywords or []))
        
            axis_start_provided = has_kw("axisStart") or (len(call.args) >= 2)
            axis_end_provided   = has_kw("axisEnd")   or (len(call.args) >= 3)
        
            if axis_start_provided or axis_end_provided:
                return node
        
            if len(call.args) >= 1:
                call.args.append(ast.Tuple(elts=[ast.Constant(0), ast.Constant(0)], ctx=ast.Load()))
                call.args.append(ast.Tuple(elts=[ast.Constant(0), ast.Constant(1)], ctx=ast.Load()))
                return node
        
            return node

        return node


def process_centering_task(task, max_centering_passes: int = 3):
    dir_name    = task.get("dir")
    file_name   = task["file"]
    script_path = Path(task["path"])
    dest_root   = Path(task["dest_root"])
    logs_path   = task["logs_path"]

    is_flat = (dir_name is None)
    script_id = Path(file_name).stem

    detail_key = script_id if is_flat else dir_name
    script_key = None if is_flat else script_id

    try:
        original_code = script_path.read_text(encoding="utf-8")

        def _exec_and_bbox(code: str):
            import numpy as np
            ns = {"np": np}
            exec(code, ns, ns)
            result = ns.get("result")
            bbox = result.val().BoundingBox()
            return bbox

        def _threshold_from_bbox(bbox):
            max_size = max(map(abs, [bbox.xmax, bbox.xmin, bbox.ymax, bbox.ymin, bbox.zmax, bbox.zmin]))
            return max_size / 20

        def _is_centered(bbox):
            c = find_center(bbox)
            size = _threshold_from_bbox(bbox)
            return (not any(abs(ci) > size for ci in c)), c, size

        cur_code = original_code
        centered_code = None

        for _ in range(max_centering_passes):
            bbox = _exec_and_bbox(cur_code)
            ok, c, _size = _is_centered(bbox)
            if ok:
                centered_code = cur_code
                break

            tree = ast.parse(cur_code)
            tree = CenteringWorkplane(c).visit(tree)
            ast.fix_missing_locations(tree)
            new_code = ast.unparse(tree)

            bbox_check = _exec_and_bbox(new_code)
            ok2, _c2, _size2 = _is_centered(bbox_check)
            if ok2:
                centered_code = new_code
                break

            cur_code = new_code

        if centered_code is None:
            log_line(logs_path, detail_key, "centering FAILED", script=script_key)
            return

        is_different = verify_centering(
            original_code=original_code,
            new_code=centered_code
        )
        if is_different:
            log_line(logs_path, detail_key, "Models are different", script=script_key)
            return

        if is_flat:
            dest_dir = dest_root
        else:
            dest_dir = dest_root / dir_name
            dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / (script_id + "_centered.py")

        if centered_code == original_code:
            save_code(str(dest_path), original_code)
            log_line(logs_path, detail_key, f"Already centered → wrote {dest_path.name}", script=script_key)
            return

        save_code(str(dest_path), centered_code)
        log_line(logs_path, detail_key, f"FINISHED → {dest_path.name}", script=script_key)
        return

    except Exception as e:
        log_line(logs_path, detail_key, f"ERROR {e}", script=script_key)
        return


def run_centering(
    centered_dir: Path,
    standardized_dir: Path,
    logs_path: Path,
    n_workers: int = 16,
    task_timeout: int = 60,
    flat: bool = False,
):
    tasks = []

    centered_dir = str(centered_dir)
    standardized_dir = str(standardized_dir)
    logs_path = str(logs_path)

    if flat:
        already = set()
        if os.path.isdir(centered_dir):
            for f in os.scandir(centered_dir):
                if f.is_file() and f.name.endswith("_centered.py"):
                    already.add(f.name)

        for f in os.scandir(standardized_dir):
            if not f.is_file():
                continue
            if not f.name.endswith(".py"):
                continue

            stem = f.name[:-3]
            final_name = f"{stem}_centered.py"
            if final_name in already:
                continue

            tasks.append({
                "file": f.name,
                "path": f.path,
                "dest_root": centered_dir,
                "logs_path": logs_path,
            })

    else:
        processed_map = {}
        if os.path.isdir(centered_dir):
            for entry in os.scandir(centered_dir):
                if not entry.is_dir():
                    continue
                dir_name = entry.name
                stems = set()
                for fe in os.scandir(entry.path):
                    if fe.is_file() and fe.name.endswith("_centered.py"):
                        stems.add(fe.name[:-len("_centered.py")])
                processed_map[dir_name] = stems

        for dir_entry in os.scandir(standardized_dir):
            if not dir_entry.is_dir():
                continue

            dir_name = dir_entry.name
            processed_stems = processed_map.get(dir_name, set())

            for fe in os.scandir(dir_entry.path):
                if not fe.is_file():
                    continue
                if not fe.name.endswith(".py"):
                    continue

                stem = fe.name[:-3]
                if stem in processed_stems:
                    continue

                tasks.append({
                    "dir": dir_name,
                    "file": fe.name,
                    "path": fe.path,
                    "dest_root": centered_dir,
                    "logs_path": logs_path,
                })

    pool = ProcessPool(
        task_func=process_centering_task,
        task_args=tasks,
        n_processes=int(n_workers),
        timeout=int(task_timeout),
    )

    unprocessed_args = pool.run()

    log_line(
        logs_path,
        "SYSTEM",
        f"centering pipeline done (tasks_total={len(tasks)}, unprocessed={len(unprocessed_args)})",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--centered-dir", type=Path, required=True)
    p.add_argument("--standardized-dir", type=Path, required=True)
    p.add_argument("--logs-path", type=Path, required=True)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--task-timeout", type=int, default=60)
    p.add_argument("--flat", action="store_true")
    a = p.parse_args()

    a.centered_dir.mkdir(parents=True, exist_ok=True)
    a.logs_path.parent.mkdir(parents=True, exist_ok=True)

    run_centering(
        centered_dir=a.centered_dir,
        standardized_dir=a.standardized_dir,
        logs_path=a.logs_path,
        n_workers=a.n_workers,
        task_timeout=a.task_timeout,
        flat=a.flat,
    )
