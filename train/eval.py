import os
import time
import shutil
import trimesh
from collections import defaultdict
from multiprocessing import Process, get_context
from multiprocessing.pool import Pool
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import sys

# font
os.environ['FONTCONFIG_PATH'] = '/etc/fonts'
os.environ['FONTCONFIG_FILE'] = '/etc/fonts/fonts.conf'

# -----------------------------------------------------------------------------
# Environment tweaks – important for head-less servers and reproducibility
# -----------------------------------------------------------------------------

os.environ["PYGLET_HEADLESS"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -----------------------------------------------------------------------------
# Helper classes – allow         Pool inside Pool (CadQuery leaks memory, so we
#                                have to sandbox every CAD exec      in its own
#                                short-lived Process).
# -----------------------------------------------------------------------------


class _NonDaemonProcess(Process):
    """A `multiprocessing.Process` that is *never* daemonised.
    Daemon processes cannot spawn children – CadQuery needs that, so we turn the
    flag off by overriding the property.
    """

    def _get_daemon(self):  # noqa: D401, D401 (property style)
        return False

    def _set_daemon(self, _value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class _NonDaemonPool(Pool):
    """`multiprocessing.Pool` that uses `_NonDaemonProcess`."""

    def Process(self, *args, **kwargs):  # noqa: N802 (match stdlib signature)
        proc = super().Process(*args, **kwargs)
        proc.__class__ = _NonDaemonProcess  # type: ignore[attr-defined]
        return proc


# -----------------------------------------------------------------------------
# Heavy imports – performed *inside* each worker so the main process stays light
# -----------------------------------------------------------------------------


def _worker_init():
    """Executed once in every worker process (fork-server context)."""
    import cadquery as _cq  # noqa: WPS433 (heavy import – intentional)
    import trimesh as _trimesh  # noqa: WPS433
    from scipy.spatial import cKDTree as _cKDTree  # noqa: WPS433

    globals().update(cq=_cq, trimesh=_trimesh, cKDTree=_cKDTree)


# -----------------------------------------------------------------------------
# Geometry & metric helpers
# -----------------------------------------------------------------------------


def _compound_to_mesh(compound: "cq.Workplane") -> "trimesh.Trimesh":  # type: ignore[name-defined]
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def _cad_code_to_mesh(code: str):
    """Execute CadQuery snippet (expects `result` variable) → `trimesh` mesh."""
    try:
        ns = {}
        exec(code, ns)
        if "result" in ns:
            result = ns["result"]
        elif "r" in ns:
            result = ns["r"]

        return _compound_to_mesh(result.val())  # type: ignore[index]
    except Exception as exc:  # noqa: BLE001
        raise(exc)


def _chamfer_distance(pred_mesh, gt_mesh, n_points: int = 8192):
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, n_points)  # type: ignore[attr-defined]
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, n_points)  # type: ignore[attr-defined]

    gt_d, _ = cKDTree(gt_pts).query(pred_pts, k=1)
    pred_d, _ = cKDTree(pred_pts).query(gt_pts, k=1)
    return np.mean(gt_d ** 2) + np.mean(pred_d ** 2)


def _iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume
        
        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except:
        pass


def _normalize_trimesh_to_unit(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    if m.vertices.size == 0:
        return m
    center = (m.bounds[0] + m.bounds[1]) / 2.0
    m.apply_translation(-center)
    extent = max(m.extents) if max(m.extents) > 1e-9 else 1.0
    m.apply_scale(1 / extent)
    m.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return m

# -----------------------------------------------------------------------------
# Metrics for a single sample – executed in a *grand-child* process, so crashes
# or CadQuery leaks cannot hurt the pool worker.
# -----------------------------------------------------------------------------

def _metrics_for_snippet(code: str, gt_path: str, n_points: int):
    base_name = os.path.basename(gt_path)
    try:
        pred_mesh = _cad_code_to_mesh(code)
    except Exception as exc:  # noqa: BLE001
        return dict(file_name=base_name, cd=None, iou=None)

    if pred_mesh is None:
        return dict(file_name=base_name, cd=None, iou=None)

    try:
        gt_mesh = trimesh.load_mesh(gt_path)  # type: ignore[attr-defined]

        pred_mesh = _normalize_trimesh_to_unit(pred_mesh)
        gt_mesh = _normalize_trimesh_to_unit(gt_mesh)
        cd_val = _chamfer_distance(pred_mesh, gt_mesh, n_points)
        iou_val = _iou(gt_mesh, pred_mesh)
        out_dict = dict(file_name=base_name, cd=cd_val, iou=iou_val)
        return out_dict
    except Exception as exc:  # noqa: BLE001
        print(f"[{base_name}] Metric computation failed: {exc}")
        return dict(file_name=base_name, cd=None, iou=None)

# -----------------------------------------------------------------------------
# Timeout-guard – isolates heavy CadQuery execution from workers.
# -----------------------------------------------------------------------------

def _run_with_timeout(args: Tuple[str, str, int], timeout: int = 60):
    ctx = get_context("fork")  # fast and safe (no CUDA here)
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(target=_child_entry, args=(child_conn, args))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        parent_conn.close()
        return "__TIMEOUT__"

    result = parent_conn.recv() if parent_conn.poll() else "__CRASH__"
    parent_conn.close()
    return result


def _child_entry(conn, args):
    try:
        conn.send(_metrics_for_snippet(*args))
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Public API – high-level helper around the pool
# -----------------------------------------------------------------------------

def compute_metrics_for_files(
    cad_files: List[str],
    gt_mesh_dir: str,
    *,
    n_points: int = 8192,
    workers: int = 8,
    timeout: int = 60,
):
    """Compute CD & IoU for each CadQuery script in *cad_files*."""
    global _POOL  # pylint: disable=global-statement

    if "_POOL" not in globals():
        ctx = get_context("forkserver")
        _POOL = _NonDaemonPool(processes=workers, context=ctx, initializer=_worker_init)

        import atexit
        atexit.register(lambda: (_POOL.close(), _POOL.join()))

    job_args = []
    for cad_path in cad_files:
        with open(cad_path, "r", encoding="utf-8") as file:
            code_str = file.read()

        gt_name = os.path.basename(cad_path).split("+")[0] + ".stl"
        gt_name = gt_name.replace('.txt', '')
        gt_path = os.path.join(gt_mesh_dir, gt_name)
        job_args.append((code_str, gt_path, n_points))

    async_res = [_POOL.apply_async(_run_with_timeout, args=(arg, timeout)) for arg in job_args]

    results = []
    for res in tqdm(async_res, total = len(cad_files), leave=False):
        out = res.get()
        if out in ("__TIMEOUT__", "__CRASH__"):
            print(f"[Worker] {out} while processing a sample – skipping.")
            results.append(dict(file_name=None, cd=None, iou=None))
        else:
            results.append(out)
    return results

# -----------------------------------------------------------------------------
# Top-level evaluation routine – matches the behaviour of the original script
# -----------------------------------------------------------------------------

def evaluate(
    gt_mesh_path: str,
    pred_py_path: str,
    best_names_path: str,
    *,
    n_points: int = 8192,
    workers: int = 8,
    timeout: int = 60,
):
    start = time.time()

    if os.path.exists(best_names_path):
        os.remove(best_names_path)
    
    cad_files = [os.path.join(pred_py_path, f) for f in os.listdir(pred_py_path) if f.endswith(".txt")]
    metrics = compute_metrics_for_files(cad_files, gt_mesh_path, n_points=n_points, workers=workers, timeout=timeout)

    grouped = defaultdict(lambda: defaultdict(list))
    for m in metrics:
        key = m["file_name"]
        if key is None:
            continue
        if m["cd"] is not None:
            grouped[key]["cd"].append(m["cd"])
        if m["iou"] is not None:
            grouped[key]["iou"].append(m["iou"])

    best_names, cd_values, iou_values = [], [], []
    ir_cd = len(metrics) - len(grouped)
    ir_iou = len(metrics) - len(grouped)
    for base, vals in grouped.items():
        if vals["cd"]:
            best_idx = int(np.argmin(vals["cd"]))
            best_names.append(f"{base}+{best_idx}.py")
            cd_values.append(vals["cd"][best_idx])
        else:
            ir_cd += 1

        if vals["iou"]:
            iou_values.append(float(np.max(vals["iou"])))
        else:
            ir_iou += 1

    elapsed = time.time() - start
    print(
        f"CD missing for {ir_cd / len(metrics):.4%}\n"
        f"IoU missing for {ir_iou / len(metrics):.4%}\n"
        f"Mean CD   : {1000*np.mean(cd_values):.6f}\n"
        f"Median CD : {1000*np.median(cd_values):.6f}\n"
        f"Mean IoU  : {np.mean(iou_values):.6f}\n"
        f"Completed in {elapsed / 60:.1f} min.")


# -----------------------------------------------------------------------------
# Script entry point – adjust the paths below to your environment.
# -----------------------------------------------------------------------------
import argparse

if __name__ == "__main__":

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    parser = argparse.ArgumentParser(description="Evaluate 3D mesh predictions")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["mcb", "fusion360", "deepcad"],
        default="deepcad",
        help="Dataset to evaluate on (mcb, fusion360, or deepcad)"
    )
    parser.add_argument(
        "--pred_py_path", 
        type=str, 
        required=True,
        help="Path to prediction files"
    )
    args = parser.parse_args()
    dataset_paths = {
        "mcb": "./MCB",
        "fusion360": "./Fusion360",
        "deepcad": "./Deepcad"
    }
    
    if args.dataset not in dataset_paths:
        raise ValueError(f"Dataset must be one of: {', '.join(dataset_paths.keys())}")
    
    gt_mesh_path = dataset_paths[args.dataset]
    evaluate(
        gt_mesh_path=gt_mesh_path,
        pred_py_path=args.pred_py_path,
        best_names_path="./best_names",

        n_points=8192,
        workers=4,
        timeout=60,
    )
