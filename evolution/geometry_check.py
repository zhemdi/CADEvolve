# geometry_check.py  ─────────────────────────────────────────────
import math
import cadquery as cq
import numpy as np

from OCP.TopAbs      import TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND, TopAbs_FACE, TopAbs_EDGE
from OCP.TopExp      import TopExp_Explorer
from OCP.TopoDS      import TopoDS_Shape
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.GProp       import GProp_GProps          # still used by CadQuery internally

# ────────────────────────────────────────────────────────────────
# low-level helpers
# ────────────────────────────────────────────────────────────────
def _subshapes(topo_shape: TopoDS_Shape, kind) -> list[TopoDS_Shape]:
    out, exp = [], TopExp_Explorer()
    exp.Init(topo_shape, kind)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def _min_distance(s1: TopoDS_Shape, s2: TopoDS_Shape) -> float:
    d = BRepExtrema_DistShapeShape(s1, s2)
    d.Perform()
    return float("inf") if not d.IsDone() else d.Value()


def _volume(shape: TopoDS_Shape) -> float:
    """Exact volume via CadQuery’s wrapper around OCCT GProp_GProps."""
    return cq.Shape.cast(shape).Volume()


# ────────────────────────────────────────────────────────────────
# public API – call this with a **TopoDS_Shape**
# ────────────────────────────────────────────────────────────────
def _check_shape(shape: TopoDS_Shape, *, name: str = "<unnamed>") -> tuple[bool, str]:
    """
    Validate a generated CAD part.

    Returns
    -------
    ok : bool
        True iff the part passes **all** tests.
    report : str
        A multi-line human-readable summary.
    """
    try:
        # ── basic metrics --------------------------------------------------
        bbox       = cq.Shape.cast(shape).BoundingBox()
        max_extent = max(bbox.xlen, bbox.ylen, bbox.zlen)
        lin_tol    = 0.05 * max_extent          # 5 % of the largest dimension
        bbox_vol   = bbox.xlen * bbox.ylen * bbox.zlen
        vol_tol    = max(1e-9, 1e-6 * bbox_vol) # absolute fallback for flat parts

        solids  = _subshapes(shape, TopAbs_SOLID)
        shells  = _subshapes(shape, TopAbs_SHELL)
        faces   = _subshapes(shape, TopAbs_FACE)
        edges   = _subshapes(shape, TopAbs_EDGE)

        n_solids, n_shells, n_faces, n_edges = (
            len(solids), len(shells), len(faces), len(edges)
        )
        top_kind = shape.ShapeType()

        # ── c1 · “exactly one solid” (or several that touch within tol) ----
        if n_solids == 0:
            c1, c1_msg = False, "✗ no solid bodies found"
        elif n_solids == 1:
            c1, c1_msg = True,  "✓ single solid body"
        else:
            # Connectivity graph: every solid must touch at least one other
            connected = True
            for i, a in enumerate(solids):
                if not any(
                    _min_distance(a, b) < lin_tol
                    for j, b in enumerate(solids) if i != j
                ):
                    connected = False
                    break
            c1 = connected
            c1_msg = (
                f"✓ {n_solids} solids but all mutually connected ≤ {lin_tol:.3g}"
                if connected else
                f"✗ {n_solids} disjoint solids (> {lin_tol:.3g})"
            )

        # ── c2 · top-level shape kind acceptable --------------------------
        c2 = top_kind in (TopAbs_SOLID, TopAbs_COMPOUND)
        c2_msg = (
            f"✓ top-level is {top_kind}"
            if c2 else
            f"✗ top-level type {top_kind} is not SOLID / COMPOUND"
        )

        # ── c3 · shell consistency (single connected boundary) ------------
        if n_shells == n_solids:
            c3 = c1            # only pass if c1 already true
            c3_msg = ("✓ boundary is a single closed shell"
                      if c3 else
                      "✗ boundary is not a single closed shell")
        else:
            c3 = False
            # more = "more" if n_shells > n_solids else "fewer"
            # c3_msg = f"✗ {n_shells} shells, {more} than {n_solids} solids"
            diff = "more" if n_shells > n_solids else "fewer"
            c3_msg = (f"✗ shell count ({n_shells}) is {diff} than solid count "
                        f"({n_solids}) – boundary disconnected")

        # # ── c4 · positive volume for every solid --------------------------
        # vols = [_volume(s) for s in solids]
        # zero_vols = [i for i, v in enumerate(vols, 1) if v <= vol_tol]
        # c4 = len(zero_vols) == 0
        # c4_msg = (
        #     "✓ all solids have positive volume"
        #     if c4 else
        #     f"✗ solids with zero/neg volume: {zero_vols}"
        # )

        # # ── c5 · no stray non-solid sub-shapes in COMPOUND ---------------
        # if top_kind == TopAbs_COMPOUND:
        #     stray_faces = n_faces - sum(len(_subshapes(s, TopAbs_FACE)) for s in solids)
        #     stray_edges = n_edges - sum(len(_subshapes(s, TopAbs_EDGE)) for s in solids)
        #     c5 = stray_faces == 0 and stray_edges == 0
        #     c5_msg = (
        #         "✓ compound contains only solids"
        #         if c5 else
        #         f"✗ compound contains {stray_faces} faces and {stray_edges} edges outside solids"
        #     )
        # else:
        #     c5, c5_msg = True, "– (not a compound)"

        ok = c1 and c2 and c3# and c4 and c5

        report = (
            f"Geometry report for {name}:\n"
            f"  c1 solid connectivity   : {c1_msg}\n"
            f"  c2 top-level type        : {c2_msg}\n"
            f"  c3 shell consistency     : {c3_msg}\n"
            # f"  c4 positive volume       : {c4_msg}\n"
            # f"  c5 stray sub-shapes      : {c5_msg}\n"
            f"  ⇒ Overall: {'PASS' if ok else 'FAIL'}"
        )
        return ok, report

    except Exception as e:
        return False, f"Exception while checking {name}: {e}"
    

# ──────────────────────────────────────────────────────────────────────────────
# public API — accepts {"name": <factory-fn>, "code": <cadquery source>}
# ──────────────────────────────────────────────────────────────────────────────
def check_geometry(part: dict) -> tuple[bool, str]:
    """
    Compile user-supplied CadQuery code, build the shape, and validate it.

    Parameters
    ----------
    part : dict
        {"name": <factory function name>, "code": <Python source code str>}

    Returns
    -------
    ok : bool
        Validation result.
    report : str
        Human-readable report.
    """
    name, code = part["name"], part["code"]

    try:
        # ── run the user code and obtain the raw TopoDS_Shape -------------
        ns: dict = {"cq": cq, "math": math}
        exec(code, ns)

        factory = ns.get(name)
        if factory is None or not callable(factory):
            raise ValueError(f'Code did not define a callable named "{name}"')

        shape = factory().val().wrapped  # cadquery Workplane → TopoDS_Shape
        return _check_shape(shape, name=name)

    except Exception as e:
        return False, f"Exception while generating {name}: {e}"