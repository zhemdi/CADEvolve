import os
import traceback
import resource
from pathlib import Path
import cadquery as cq
import argparse


def _bbox_checks(shape_or_obj, extent_min=195.0, extent_max=205.0, center_tol=5.0):
    """
    Returns (ok: bool, msg: str). ok==True iff max extent ∈ [extent_min, extent_max]
    and center ∈ [-center_tol, center_tol]^3.
    """
    try:
        bb = shape_or_obj.BoundingBox()
    except Exception as e:
        return False, f"BoundingBox() failed: {e}"

    cx = 0.5 * (bb.xmin + bb.xmax)
    cy = 0.5 * (bb.ymin + bb.ymax)
    cz = 0.5 * (bb.zmin + bb.zmax)
    m  = max(bb.xlen, bb.ylen, bb.zlen)

    if not (extent_min <= m <= extent_max):
        return False, f"max_extent={m:.3f} not in [{extent_min},{extent_max}]"
    if not (-center_tol <= cx <= center_tol and
            -center_tol <= cy <= center_tol and
            -center_tol <= cz <= center_tol):
        return False, f"center=({cx:.3f},{cy:.3f},{cz:.3f}) out of [-{center_tol},{center_tol}]^3"
    return True, "OK"


def py_file_to_mesh_file(py_path, out_mesh_dir):
    base = os.path.splitext(os.path.basename(py_path))[0]
    mesh_path = os.path.join(out_mesh_dir, f"{base}.stl")

    try:
        if os.path.isfile(mesh_path) and os.path.getsize(mesh_path) > 0:
            return
    except Exception as e:
        raise(e)
        #pass

    try:
        with open(py_path, 'r', encoding='utf-8', errors='ignore') as f:
            py_string = f.read()

        ns = {}
        exec(py_string, ns)

        obj = ns.get('result')
        if obj is None:
            raise KeyError("Result not found")

        try:
            shape_or_obj = obj.val()
        except AttributeError:
            shape_or_obj = obj

        ok, msg = _bbox_checks(shape_or_obj, extent_min=195.0, extent_max=205.0, center_tol=5.0)
        if not ok:
            print(f"[SKIP QA] {base}: {msg}")
            return

        cq.exporters.export(shape_or_obj, mesh_path, tolerance=0.001, angularTolerance=0.1)

    except Exception as e:
        raise(e)
        #return