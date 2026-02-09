import os, ast, resource, argparse, math
import numpy as np
from pathlib import Path
from typing import Set, Dict
from utils.canonicalization.helpers import ProcessPool, log_line, save_code, tessellate_solid, find_center

TARGET_METHODS = {
    'Workplane', 'workplane', 'workplaneFromTagged', 'transformed', 'center', 'Plane', 'Vector',

    'moveTo', 'lineTo', 'vLine', 'hLine', 'vLineTo', 'hLineTo', 'polarLine',
    'polyline', 'spline', 'threePointArc', 'radiusArc', 'tangentArc', 'ellipseArc',
    'rect', 'circle', 'ellipse', 'polygon', 'offset2D', 'slot2D',

    'pushPoints', 'rarray', 'polarArray', 'splineApprox',

    'extrude', 'revolve', 'loft', 'sweep',

    'box', 'wedge', 'cylinder', 'cone', 'sphere', 'torus', 'trapezoid',

    'shell', 'thicken', 'offset', 'fillet', 'chamfer',
}


def is_target_call(call: ast.Call) -> bool:
    f = call.func
    return isinstance(f, ast.Attribute) and f.attr in TARGET_METHODS


class RoundConstants(ast.NodeTransformer):
    def __init__(self, ndigits: int = 3, int_tol: float = 1e-9):
        self.ndigits = int(ndigits)
        self.int_tol = float(int_tol)

    def _round_number(self, x):
        if isinstance(x, bool):
            return x

        if isinstance(x, int):
            return x

        if isinstance(x, float):
            if not math.isfinite(x):
                return x

            y = round(x, self.ndigits)

            yi = int(round(y))
            if abs(y - yi) <= self.int_tol:
                return yi

            if y == 0:
                return 0.0

            return y

        return x

    def visit_Constant(self, node: ast.Constant):
        v = node.value
        if isinstance(v, (bool, int, float)):
            new_v = self._round_number(v)
            if new_v is v:
                return node
            return ast.copy_location(ast.Constant(new_v), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
            v = node.operand.value
            if isinstance(v, bool):
                return node
            if isinstance(v, (int, float)):
                signed = (-1 if isinstance(node.op, ast.USub) else 1) * v
                new_v = self._round_number(signed)
                return ast.copy_location(ast.Constant(new_v), node)
        return self.generic_visit(node)


class ScaleNumsInCalls(ast.NodeTransformer):
    def __init__(self, scale: float):
        self.scale = scale

        self.ANGLE_KW: Set[str] = {
            "angle", "angleDegrees",
            "angle1", "angle2",
            "startAngle", "endAngle",
            "rotationAngle",
            "twistAngle", "twistAngleDegrees",
        }

        self.NON_SCALING_KW_BY_FUNC: Dict[str, Set[str]] = {
            "polygon": {"nSides"},
            "regularPolygon": {"n"},
        
            "rarray": {"xCount", "yCount"},
        
            "polarArray": {"count", "startAngle", "angle"},
        
            "revolve": {"angleDegrees", "angle"},
            "rotate": {"angleDegrees", "angle"},
            "twistExtrude": {"angleDegrees", "twistAngle", "twistAngleDegrees"},
        
            "makeSplineApprox": {"minDeg", "maxDeg"},
            "splineApprox": {"minDeg", "maxDeg"},
        
            "parametricCurve": {"N"},
            "parametricSurface": {"N", "minDeg", "maxDeg"},
        
            "transformed": {"rotate"},
            
            "ellipseArc": {"angle1", "angle2", "startAngle", "endAngle"},
            "cskHole": {"cskAngle"}
        }

        self.NON_SCALING_POS_BY_FUNC: Dict[str, Set[int]] = {
            "polygon": {0},             # nSides
            "regularPolygon": {1},      # n
        
            "rarray": {2, 3},           # xCount, yCount
        
            # polarArray(radius, startAngle, angle, count, ...)
            "polarArray": {1, 2, 3},    # startAngle, angle, count
        
            # polarLine(length, angle)
            "polarLine": {1},           # angle
        
            # revolve(angleDegrees, ...)
            "revolve": {0},             # angleDegrees
        
            # rotate(axisStart, axisEnd, angleDegrees)
            "rotate": {2},              # angleDegrees
        
            # twistExtrude(distance, angleDegrees/...)
            "twistExtrude": {1},        # angleDegrees / twistAngle
        
            # ellipseArc(xRadius, yRadius, angle1, angle2, ...)
            "ellipseArc": {2, 3},       # angle1, angle2
        
            # splineApprox(points, tol, minDeg, maxDeg)
            "splineApprox": {2, 3},     # minDeg, maxDeg
            "cskHole": {2}
        }

    def _scale_const(self, node: ast.AST, value):
        if isinstance(value, bool):
            return ast.copy_location(ast.Constant(value), node)
        if isinstance(value, (float, int)):
            v = float(value * self.scale)
            return ast.copy_location(ast.Constant(v), node)
        return node

    def _scale_any(self, n: ast.AST):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return self._scale_const(n, n.value)

        if (isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd))
            and isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, (int, float))):
            signed = (-1 if isinstance(n.op, ast.USub) else 1) * n.operand.value
            return self._scale_const(n, signed)

        if isinstance(n, ast.List):
            n.elts = [self._scale_any(e) for e in n.elts]
            return n

        if isinstance(n, ast.Tuple):
            n.elts = [self._scale_any(e) for e in n.elts]
            return n

        if isinstance(n, ast.Dict):
            n.keys  = [self._scale_any(e) for e in n.keys]
            n.values= [self._scale_any(e) for e in n.values]
            return n

        if isinstance(n, ast.Call):
            return self._scale_call(n)

        return n

    def _scale_call(self, call: ast.Call) -> ast.Call:
        f = call.func
        if isinstance(f, ast.Attribute):
            func_name = f.attr
        elif isinstance(f, ast.Name):
            func_name = f.id
        else:
            func_name = ""

        new_args = []
        skip_pos = self.NON_SCALING_POS_BY_FUNC.get(func_name, set())
        for i, a in enumerate(call.args):
            if i in skip_pos:
                new_args.append(a)
            else:
                new_args.append(self._scale_any(a))
        call.args = new_args

        skip_kw = self.NON_SCALING_KW_BY_FUNC.get(func_name, set())
        for kw in call.keywords:
            if kw.arg is None:
                kw.value = self._scale_any(kw.value)
                continue

            if kw.arg in skip_kw or kw.arg in self.ANGLE_KW:
                continue
            kw.value = self._scale_any(kw.value)

        return call

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Call) and is_target_call(node.value):
            node.value = self._scale_call(node.value)
            return node

        node.value = self._scale_any(node.value)
        return node


def as_shape(obj):
    if hasattr(obj, "val") and callable(obj.val):
        return obj.val()
    return obj


def _bbox_of(obj):
    return as_shape(obj).BoundingBox()


def check_code(code: str): 
    ns = {"np": np}
    exec(code, ns, ns)
    result = ns.get('result')

    bbox = _bbox_of(result)
    c = find_center(bbox)
    max_size = max(map(abs, [bbox.xmax, bbox.xmin, bbox.ymax, bbox.ymin, bbox.zmax, bbox.zmin]))
    
    return c, max_size


def verify_scaling(original_code: str, new_code: str, scale_factor: float) -> bool:

    import trimesh
    from scipy.spatial import cKDTree
    import gc

    ns_orig = {"np": np}
    exec(original_code, ns_orig, ns_orig)
    result_orig = ns_orig.get('result')

    bbox_orig = _bbox_of(result_orig)
    rotation_vector = find_center(bbox_orig)

    ns_new = {"np": np}
    exec(new_code, ns_new, ns_new)
    result_new = ns_new.get('result')

    shape_new  = as_shape(result_new)
    shape_orig = as_shape(result_orig)

    result_rolled_back = shape_new.scale(1 / scale_factor).translate(rotation_vector)
    
    vertices_orig, triangles_orig = tessellate_solid(as_shape(shape_orig).wrapped)
    mesh_orig = trimesh.Trimesh(vertices=vertices_orig, faces=triangles_orig, process=False, validate=False)

    vertices_rolled_back, triangles_rolled_back = tessellate_solid(as_shape(result_rolled_back).wrapped)
    mesh_rolled_back = trimesh.Trimesh(vertices=vertices_rolled_back, faces=triangles_rolled_back, process=False, validate=False)

    extent_orig = mesh_orig.extents.max()
    extent_rolled_back = mesh_rolled_back.extents.max()

    mesh_rolled_back.apply_transform(trimesh.transformations.scale_matrix(1 / extent_rolled_back))
    mesh_rolled_back.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

    mesh_orig.apply_transform(trimesh.transformations.scale_matrix(1 / extent_orig))
    mesh_orig.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

    n_points = 20_000
    gt_points, _ = trimesh.sample.sample_surface(mesh_orig, n_points)
    pred_points, _ = trimesh.sample.sample_surface(mesh_rolled_back, n_points)

    gt_points = gt_points.astype(np.float32, copy=False)
    pred_points = pred_points.astype(np.float32, copy=False)

    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)

    gt_distance = gt_distance.astype(np.float32, copy=False)
    pred_distance = pred_distance.astype(np.float32, copy=False)

    cd = np.mean(gt_distance * gt_distance) + np.mean(pred_distance * pred_distance)

    del mesh_orig, mesh_rolled_back, vertices_orig, triangles_orig, vertices_rolled_back, triangles_rolled_back
    del gt_points, pred_points, gt_distance, pred_distance
    gc.collect()

    return cd > 0.15


def process_scaling_task(task, delta: float = 2.5):
    dir_name    = task.get("dir")
    file_name   = task["file"]
    script_path = Path(task["path"])
    orig_path   = Path(task["orig_path"])
    dest_root   = Path(task["dest_root"])
    log_path    = task["log_path"]

    is_flat = (dir_name is None)
    detail_key = (Path(file_name).stem if is_flat else dir_name)
    script_key = (None if is_flat else Path(file_name).stem)

    def _round_half_up(x: float) -> int:
        return int(math.floor(x + 0.5))

    try:
        centered_code = script_path.read_text(encoding="utf-8")
        c_orig, max_size_orig = check_code(centered_code)

        threshold_center = max_size_orig / 20.0
        if any(abs(ci) > threshold_center for ci in c_orig):
            log_line(
                log_path,
                detail_key,
                f"{file_name}: Center of the original model is wrong, c={c_orig}, tol={threshold_center}, max_size={max_size_orig}",
                script=script_key,
            )
            return

        if (100 - delta) <= max_size_orig <= (100 + delta):
            if is_flat:
                dest_dir = dest_root
            else:
                dest_dir = dest_root / dir_name
                dest_dir.mkdir(parents=True, exist_ok=True)

            dest_path = dest_dir / (Path(file_name).stem + "_scaled.py")
            save_code(str(dest_path), centered_code)

            log_line(
                log_path,
                detail_key,
                f"{file_name}: Already scaled → wrote {dest_path.name}",
                script=script_key,
            )
            return

        def _attempt(scale_den: float, tag: str):
            scale_factor = 100.0 / float(scale_den)

            tree = ast.parse(centered_code)
            tree = RoundConstants(ndigits=3).visit(tree)
            ast.fix_missing_locations(tree)

            tree = ScaleNumsInCalls(scale_factor).visit(tree)
            ast.fix_missing_locations(tree)
            new_code = ast.unparse(tree)

            c_new, max_size_new = check_code(new_code)

            ok_center = not any(abs(ci) > 5.0 + 1e-3 for ci in c_new)
            ok_size   = (100 - delta) <= max_size_new <= (100 + delta)

            return {
                "ok": (ok_center and ok_size),
                "tag": tag,
                "scale_den": float(scale_den),
                "scale_factor": float(scale_factor),
                "new_code": new_code,
                "c_new": c_new,
                "max_size_new": max_size_new,
            }

        r1 = _attempt(scale_den=max_size_orig, tag="raw")

        r_final = r1
        if not r1["ok"]:
            den2 = _round_half_up(max_size_orig)

            r2 = _attempt(scale_den=den2, tag="rounded_den")
            r_final = r2

            if not r2["ok"]:
                log_line(
                    log_path,
                    detail_key,
                    f"{file_name}: scaling FAILED. "
                    f"try1(tag={r1['tag']}, den={r1['scale_den']}, s={r1['scale_factor']:.6g}, c={r1['c_new']}, max_size_new={r1['max_size_new']:.6g}); "
                    f"try2(tag={r2['tag']}, den={r2['scale_den']}, s={r2['scale_factor']:.6g}, c={r2['c_new']}, max_size_new={r2['max_size_new']:.6g})",
                    script=script_key,
                )
                return

        new_code = r_final["new_code"]
        scale_factor = r_final["scale_factor"]

        original_emitted_code = orig_path.read_text(encoding="utf-8")
        is_different = verify_scaling(
            original_code=original_emitted_code,
            new_code=new_code,
            scale_factor=scale_factor
        )
        if is_different:
            log_line(
                log_path,
                detail_key,
                f"{file_name}: Models are different",
                script=script_key,
            )
            return

        if is_flat:
            dest_dir = dest_root
        else:
            dest_dir = dest_root / dir_name
            dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / (Path(file_name).stem + "_scaled.py")
        save_code(str(dest_path), new_code)

        log_line(
            log_path,
            detail_key,
            f"{file_name}: FINISHED({r_final['tag']}) → {dest_path.name}",
            script=script_key,
        )
        return

    except Exception as e:
        log_line(
            log_path,
            detail_key,
            f"{file_name}: ERROR {repr(e)}",
            script=script_key,
        )
        return


def run_scaling(
    scaled_dir: Path,
    centered_dir: Path,
    logs_path: Path,
    standardized_dir: Path,
    n_workers: int,
    task_timeout: int,
    flat: bool = False
):
    scaled_dir = str(scaled_dir)
    centered_dir = str(centered_dir)
    standardized_dir = str(standardized_dir)
    logs_path = str(logs_path)

    tasks = []

    if flat:
        already = set()
        if os.path.isdir(scaled_dir):
            for f in os.scandir(scaled_dir):
                if f.is_file() and f.name.endswith("_scaled.py"):
                    already.add(f.name)

        for f in os.scandir(centered_dir):
            if not f.is_file():
                continue
            if not f.name.endswith("_centered.py"):
                continue

            stem = f.name[:-3]
            final_name = f"{stem}_scaled.py"
            if final_name in already:
                continue

            orig_name = f.name[:-len("_centered.py")] + ".py"
            orig_path = os.path.join(standardized_dir, orig_name)

            tasks.append({
                "file": f.name,
                "path": f.path,
                "orig_path": orig_path,
                "dest_root": scaled_dir,
                "log_path": logs_path,
            })

    else:
        processed_map: dict[str, set[str]] = {}
        if os.path.isdir(scaled_dir):
            for entry in os.scandir(scaled_dir):
                if not entry.is_dir():
                    continue

                dir_name = entry.name
                stems = set()
                for fe in os.scandir(entry.path):
                    if fe.is_file() and fe.name.endswith("_scaled.py"):
                        stems.add(fe.name[:-len("_scaled.py")])
                processed_map[dir_name] = stems

        if os.path.isdir(centered_dir):
            for dir_entry in os.scandir(centered_dir):
                if not dir_entry.is_dir():
                    continue

                dir_name = dir_entry.name
                existing_scaled = processed_map.get(dir_name, set())

                for file_entry in os.scandir(dir_entry.path):
                    if not file_entry.is_file():
                        continue
                    if not file_entry.name.endswith("_centered.py"):
                        continue

                    out_stem = file_entry.name[:-len(".py")]
                    if out_stem in existing_scaled:
                        continue

                    orig_name = file_entry.name[:-len("_centered.py")] + ".py"
                    orig_path = os.path.join(standardized_dir, dir_name, orig_name)

                    if not (os.path.isfile(orig_path) and orig_path.endswith(".py")):
                        log_line(
                            logs_path,
                            dir_name,
                            f"{file_entry.name}: SKIP — no original emitted file at {orig_path}",
                            script=Path(file_entry.name).stem,
                        )
                        continue

                    tasks.append({
                        "dir": dir_name,
                        "file": file_entry.name,
                        "path": file_entry.path,
                        "orig_path": orig_path,
                        "dest_root": scaled_dir,
                        "log_path": logs_path,
                    })

    pool = ProcessPool(
        task_func=process_scaling_task,
        task_args=tasks,
        n_processes=n_workers,
        timeout=task_timeout,
    )

    unprocessed_args = pool.run()

    log_line(
        logs_path,
        "SYSTEM",
        f"scaling pipeline done (tasks_total={len(tasks)}, unprocessed={len(unprocessed_args)})",
    )


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument('--scaled-dir', type=Path, required=True)
    p.add_argument('--centered-dir', type=Path, required=True)
    p.add_argument("--logs-path", type=Path, required=True)
    p.add_argument("--standardized-dir", type=Path, required=True)
    p.add_argument("--system-reserved-ratio", type=float, default=0.1)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--task-timeout", type=int, default=60)
    p.add_argument("--flat", action="store_true")
    a = p.parse_args()

    scaled_dir, centered_dir, logs_path, standardized_dir, system_reserved_ratio, n_workers, task_timeout = (
        a.scaled_dir, a.centered_dir, a.logs_path, a.standardized_dir, a.system_reserved_ratio, a.n_workers, a.task_timeout
    )

    scaled_dir.mkdir(exist_ok=True, parents=True)
    logs_path.parent.mkdir(exist_ok=True, parents=True)

    run_scaling(
        scaled_dir=scaled_dir,
        centered_dir=centered_dir,
        logs_path=logs_path,
        standardized_dir=standardized_dir,
        n_workers=n_workers,
        task_timeout=task_timeout,
        flat=a.flat
    )