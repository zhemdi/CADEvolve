from __future__ import annotations

# ────────────────────────────────────────────────────────────────
# Original imports
# ────────────────────────────────────────────────────────────────
import json
import os
os.environ.setdefault("FONTCONFIG_FILE", "/etc/fonts/fonts.conf")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")
import re
import traceback
import tempfile
import uuid
import base64
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from functools import partial
from geometry_check import check_geometry

import numpy as np
import cadquery as cq  

from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures as cf



# choose ONE of the two – both solve the problem
# mp.set_start_method("spawn",      force=True)    # independent interpreter

# ────────────────────────────────────────────────────────────────
# rendering & image‑processing deps (unchanged from reference)
# ────────────────────────────────────────────────────────────────
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.transform import resize
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopAbs      import TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_SHELL

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from multiprocessing import Process
from multiprocessing.pool import Pool

N_WORKERS = min(os.cpu_count(), 64)


from openai import OpenAI





# ────────────────────────────────────────────────────────────────
# Configuration (unchanged where possible, new items marked ★)
# ────────────────────────────────────────────────────────────────
ROOT              = Path(__file__).resolve().parent
DB_PATH           = ROOT / "code_db.json"
EMBED_DIR         = ROOT / "embeddings"      # each embedding → *.npy*
TMP_RENDER_DIR    = ROOT / "tmp_render"      # ★ all intermediate .png/.stl/.step go here

EMBED_MODEL       = "text-embedding-3-small"
LLM_MODEL         = "gpt5-mini"               # chat + code model (same for vision requests)
TOP_K             = 7                        # similar parts to retrieve
SHAPES_PER_LEVEL  = 10                      # new shapes requested per level
MAX_LEVELS        = 1                       # how many complexity levels to add
MAX_REFINES       = 3                        # compile‑error fixing attempts
MAX_VISUAL_REFINES = 4                       # ★ attempts to satisfy visual validator
MAX_MODEL_RETRIES  = 5                       # ★ internal retry loop when calling OpenAI

API_KEY     = os.getenv("OPENAI_API_KEY","api_key")

# ── Instantiate clients once ────────────────────────────────────
client = OpenAI(api_key=API_KEY)



TMP_RENDER_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "cadquery_examples.txt", "r") as f:
    cq_site_examples = f.read()

# effort value reused by propose_shapes & chat
effort = "high"

TIMEOUT_STEP = 60

class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        proc = super(NonDaemonPool, self).Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc

# ────────────────────────────────────────────────────────────────
# Embedding helpers 
# ────────────────────────────────────────────────────────────────

def save_embedding(name: str, emb: List[float]) -> str:
    path = EMBED_DIR / f"{name}.npy"
    np.save(path, np.asarray(emb, dtype=np.float32))
    return str(path)


def load_embedding(path: str) -> np.ndarray:
    return np.load(path)


def get_embedding(client, text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

# ────────────────────────────────────────────────────────────────
# Database helpers 
# ────────────────────────────────────────────────────────────────

def load_db() -> List[Dict]:
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text())
    return []


def save_db(db: List[Dict]):
    DB_PATH.write_text(json.dumps(db, indent=2))

# ────────────────────────────────────────────────────────────────
# Similarity search 
# ────────────────────────────────────────────────────────────────

def similarity_search(client, db: List[Dict], query: str, k: int = TOP_K) -> List[Dict]:
    q_emb = np.asarray(get_embedding(client, query))[None, :]
    emb_matrix = np.vstack([load_embedding(p["embedding_path"]) for p in db])
    sims = cosine_similarity(q_emb, emb_matrix)[0]
    return [p for p, _ in sorted(zip(db, sims), key=lambda x: -x[1])[:k]]

# ────────────────────────────────────────────────────────────────
# Seed – level‑0 primitives 
# ────────────────────────────────────────────────────────────────


def seed_primitives(client, db: List[Dict]):
    if db:
        return  # already seeded

    primitives = [
        # ── Basic solids ───────────────────────────────────────
        {
            "name": "box_prism",
            "abstract": "rectangular prism",
            "detailed": "Box L×W×H centred at origin (extrude rectangle).",
            "code": (
                "def box_prism(L=10, W=10, H=10):\n"
                "    return cq.Workplane('XY').box(L, W, H)"
            ),
        },
        {
            "name": "cylinder_basic",
            "abstract": "circular cylinder",
            "detailed": "Cylinder radius R, height H, axis Z (extrude circle).",
            "code": (
                "def cylinder_basic(R=5, H=20):\n"
                "    return cq.Workplane('XY').circle(R).extrude(H)"
            ),
        },
        {
            "name": "sphere_basic",
            "abstract": "sphere",
            "detailed": "Sphere radius R centred at origin (revolve semicircle).",
            "code": (
                "def sphere_basic(R=10):\n"
                "    return cq.Workplane('XZ').sphere(R)"
            ),
        },
        # ── Revolve example ────────────────────────────────────
        {
            "name": "cone_revolve",
            "abstract": "right circular cone",
            "detailed": "Cone base radius R, height H using 360° revolve of triangle.",
            "code": (
                "def cone_revolve(R=10, H=20):\n"
                "    return (cq.Workplane('XZ')\n"
                "            .moveTo(0, 0)\n"
                "            .lineTo(0, H)\n"
                "            .lineTo(R, 0)\n"
                "            .close()\n"
                "            .revolve(360) )"
            ),
        },
        # ── Sweep (torus‑like pipe) ────────────────────────────
        {
            "name": "pipe_sweep",
            "abstract": "straight pipe",
            "detailed": "Sweeps a circular profile of radius r along straight path L.",
            "code": (
                "def pipe_sweep(L=50, r=5):\n"
                "    path = cq.Workplane('XY').line(L, 0)\n"
                "    return (cq.Workplane('YZ')\n"
                "            .circle(r)\n"
                "            .sweep(path, multisection=False))"
            ),
        },
        # ── Loft between profiles ──────────────────────────────
        {
            "name": "loft_cylinder_to_cone",
            "abstract": "loft cylinder→cone",
            "detailed": "Lofts from circle radius R1 to radius R2 over height H.",
            "code": (
                "def loft_cylinder_to_cone(R1=10, R2=2, H=30):\n"
                "    wp = cq.Workplane('XY')\n"
                "    return (wp.circle(R1)\n"
                "              .workplane(offset=H).circle(R2)\n"
                "              .loft())"
            ),
        },
        # ── Shell example ──────────────────────────────────────
        {
            "name": "hollow_box_shell",
            "abstract": "hollow box (shell)",
            "detailed": "Creates thin‑walled box L×W×H with thickness t using shell.",
            "code": (
                "def hollow_box_shell(L=30, W=20, H=15, t=2):\n"
                "    solid = cq.Workplane('XY').box(L, W, H)\n"
                "    return solid.shell(-t)"
            ),
        },

        # ── Polygon-based prisms ──────────────────────────────────────
        {
            "name": "hex_prism",
            "abstract": "hexagonal prism",
            "detailed": "Regular-hexagon base edge-length A extruded height H.",
            "code": (
                "def hex_prism(A=10, H=20):\n"
                "    return cq.Workplane('XY').polygon(6, 2*A*math.sqrt(3)).extrude(H)"
            ),
        },
        {
            "name": "triangular_prism",
            "abstract": "equilateral triangular prism",
            "detailed": "Equilateral-triangle base edge-length A extruded height H.",
            "code": (
                "def triangular_prism(A=10, H=25):\n"
                "    h = A * math.sqrt(3)  # triangle height\n"
                "    return (cq.Workplane('XY')\n"
                "              .polyline([(0,0), (A,0), (A/2,h)]).close()\n"
                "              .extrude(H))"
            ),
        },

        # ── Revolve / sweep variants ──────────────────────────────────
        {
            "name": "torus_revolve",
            "abstract": "torus (revolve)",
            "detailed": "Torus with major-radius R and tube-radius r by 360° revolve.",
            "code": (
                "def torus_revolve(R=20, r=5):\n"
                "    return (cq.Workplane('XZ')\n"
                "              .moveTo(R, 0)\n"
                "              .circle(r)\n"
                "              .revolve(360, (0, 0, 0), (0, 1, 0)))"
            ),
        },
        {
            "name": "half_cylinder",
            "abstract": "half-cylinder",
            "detailed": "Semicircular profile radius R extruded length L.",
            "code": (
                "def half_cylinder(R=10, L=30):\n"
                "    return (cq.Workplane('XY')\n"
                "              .center(0, -R)\n"
                "              .moveTo(-R, 0)\n"
                "              .lineTo(R, 0)\n"
                "              .threePointArc((0, 2*R), (-R, 0))\n"
                "              .close()\n"
                "              .extrude(L))"
            ),
        },

        # ── Taper / loft examples ─────────────────────────────────────
        {
            "name": "frustum_cone",
            "abstract": "truncated cone (frustum)",
            "detailed": "Lofts circle radius R1 to radius R2 over height H.",
            "code": (
                "def frustum_cone(R1=12, R2=6, H=25):\n"
                "    return (cq.Workplane('XY')\n"
                "              .circle(R1)\n"
                "              .workplane(offset=H).circle(R2)\n"
                "              .loft())"
            ),
        },
        {
            "name": "square_pyramid",
            "abstract": "square pyramid",
            "detailed": "Lofts square side S to a point height H (apex).",
            "code": (
                "def square_pyramid(S=20, H=18):\n"
                "    base = cq.Workplane('XY').rect(S, S)\n"
                "    apex = base.workplane(offset=H)  \n"
                "               .pushPoints([(0, 0)])  \n"
                "               .circle(1e-6)\n"
                "    return apex.loft(combine=True)"
            ),
        },

        # ── Boolean cavity example ────────────────────────────────────
        {
            "name": "annular_ring",
            "abstract": "annular ring",
            "detailed": "Extrudes an annulus (outer radius R, thickness t) height H.",
            "code": (
                "def annular_ring(R=20, t=4, H=5):\n"
                "    return (cq.Workplane('XY')\n"
                "              .circle(R)\n"
                "              .circle(R - t)\n"
                "              .extrude(H))"
            ),
        },
        {
            "name": "wedge_basic",
            "abstract": "rectangular wedge",
            "detailed": "Block L×W×H cut by a plane inclined θ about Y-axis.",
            "code": (
                "import math\n"
                "def wedge_basic(L=30, W=20, H=15, theta=15):\n"
                "    base = cq.Workplane('XY').box(L, W, H)\n"
                "    cut = (cq.Workplane('XZ')\n"
                "             .transformed(offset=(0,0,H/2), rotate=(0,theta,0))\n"
                "             .rect(L*2, H*2)\n"
                "             .extrude(W, both=True))\n"
                "    return base.cut(cut)"
            ),
        },
                # ── Polygon & star prisms ─────────────────────────────────────
        {
            "name": "oct_prism",
            "abstract": "octagonal prism",
            "detailed": "Regular octagon edge-length A extruded height H.",
            "code": (
                "def oct_prism(A=8, H=20):\n"
                "    return cq.Workplane('XY').polygon(8, A / math.sin(math.pi / 8)).extrude(H)"
            ),
        },
        {
            "name": "star_prism",
            "abstract": "five-point star prism",
            "detailed": "2-D 5-point star outer-radius R extruded height H.",
            "code": (
                "import math\n"
                "def star_prism(R=15, H=10):\n"
                "    pts = []\n"
                "    for i in range(10):\n"
                "        ang = math.pi/5 * i\n"
                "        r   = R if i % 2 == 0 else R*0.382\n"
                "        pts.append((r*math.cos(ang), r*math.sin(ang)))\n"
                "    return cq.Workplane('XY').polyline(pts).close().extrude(H)"
            ),
        },

        # ── Revolve & toroidal variations ─────────────────────────────
        {
            "name": "quarter_torus",
            "abstract": "90-degree torus segment",
            "detailed": "Quarter-segment torus major-radius R and tube-radius r.",
            "code": (
                "def quarter_torus(R=25, r=6):\n"
                "    p = cq.Workplane('XZ').moveTo(R, 0).circle(r)\n"
                "    return p.revolve(90)"
            ),
        },
        {
            "name": "dome_hemisphere",
            "abstract": "hemispherical dome",
            "detailed": "Half-sphere radius R (upper hemisphere).",
            "code": (
                "def dome_hemisphere(R=12):\n"
                "    return (cq.Workplane('XZ')\n"
                "              .moveTo(-R, 0)\n"
                "              .threePointArc((0, R), (R, 0))\n"
                "              .close()\n"
                "              .revolve(180, (0,0,0), (1,0,0)))"
            ),
        },

        # ── Helix / sweep examples ────────────────────────────────────
        {
            "name": "helix_wire",
            "abstract": "cylindrical helix wire",
            "detailed": "Helical wire of radius R, pitch P, turns N, wire-radius r.",
            "code": (
                "def helix_wire(R=10, P=12, N=3, r=1):\n"
                "    h = P * N\n"
                "    helix = cq.Workplane('XY').parametricCurve(\n"
                "        lambda t: (R*math.cos(2*math.pi*N*t), R*math.sin(2*math.pi*N*t), h*t))\n"
                "    return cq.Workplane('XZ').moveTo(R, 0).circle(r).sweep(helix)"
            ),
        },

        # ── Loft / pyramid variants ───────────────────────────────────
        {
            "name": "pent_pyramid",
            "abstract": "pentagonal pyramid",
            "detailed": "Lofts regular pentagon side S to apex height H.",
            "code": (
                "def pent_pyramid(S=16, H=20):\n"
                "    wp = cq.Workplane('XY')\n"
                "    return (wp.polygon(5, S / (2*math.sin(math.pi/5)))\n"
                "              .workplane(offset=H)\n"
                "              .circle(0.001)\n"
                "              .loft(combine=True))"
            ),
        },

        # ── Boolean & edge-treatment demos ────────────────────────────
        {
            "name": "chamfered_box",
            "abstract": "box with uniform chamfer",
            "detailed": "Box L×W×H with all edges chamfer distance c.",
            "code": (
                "def chamfered_box(L=20, W=15, H=10, c=1.5):\n"
                "    solid = cq.Workplane('XY').box(L, W, H)\n"
                "    return solid.edges().chamfer(c)"
            ),
        },
        {
            "name": "sphere_cap",
            "abstract": "spherical cap",
            "detailed": "Cap height h cut from sphere radius R (top segment).",
            "code": (
                "def sphere_cap(R=12, h=4):\n"
                "    sphere = cq.Workplane('XY').sphere(R)\n"
                "    cutter = cq.Workplane('XY').workplane(offset=R-h).box(2*R,2*R,2*R)\n"
                "    return sphere.intersect(cutter)"
            ),
        },
        {
            "name": "spur_gear",
            "abstract": "spur gear",
            "detailed": "Simple spur gear with a specified number of teeth (n_teeth), module, and face thickness. The gear blank is a cylinder at the pitch radius; each tooth is a rectangular prism from root to addendum height and is polar-arrayed around the pitch circle, then unioned with the blank.",
            "code": (
                "def spur_gear(n_teeth=16, module=2.0, thickness=6.0):\n"
                "    pitch_diameter = module * n_teeth\n"
                "    r_pitch = pitch_diameter / 2.0\n"
                "    addendum = module\n"
                "    dedendum = 1.25 * module\n"
                "    r_root = r_pitch - dedendum\n"
                "    r_outer = r_pitch + addendum\n"
                "    circular_pitch = math.pi * module\n"
                "    half_tooth = circular_pitch / 4.0\n"
                "    gear_blank = cq.Workplane('XY').circle(r_pitch).extrude(thickness)\n"
                "    tooth = (\n"
                "        cq.Workplane('XY')\n"
                "          .polyline([\n"
                "              (r_root, -half_tooth),\n"
                "              (r_outer, -half_tooth),\n"
                "              (r_outer,  half_tooth),\n"
                "              (r_root,  half_tooth),\n"
                "          ])\n"
                "          .close()\n"
                "          .extrude(thickness)\n"
                "    )\n"
                "    gear = gear_blank\n"
                "    for i in range(n_teeth):\n"
                "        angle = 360.0 * i / n_teeth\n"
                "        rotated_tooth = tooth.rotate((0, 0, 0), (0, 0, 1), angle)\n"
                "        gear = gear.union(rotated_tooth)\n"
                "    return gear"
            ),
        },
    ]

    for part in primitives:
        emb = get_embedding(client, part["abstract"])
        part["embedding_path"] = save_embedding(part["name"], emb)
        db.append(part)

# ────────────────────────────────────────────────────────────────
# LLM helper (unchanged)
# ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a senior mechanical CAD engineer. "
    "Focus on generating **self‑contained, parametric** CadQuery v2 functions. "
    "**Do NOT call other functions – copy any logic you need.** "
    "Return pure Python (no markdown fences, without snippet) unless JSON is explicitly requested."
    f"Examples from the cadquery website:\n{cq_site_examples}" 
    # "The function must return exactly ONE solid body; overlapping or disjoint solids, loose shells or open surfaces are invalid."
    "The function must return exactly ONE solid body whose boundary is a single closed, connected shell; overlapping or disjoint solids, loose shells or open surfaces are invalid."
)


def chat(client, msgs: List[Dict[str, Any]], **kw) -> str:
    for msg in msgs:
        if msg["role"] == "system":
            break
    else:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})
    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=msgs,
        **kw,
    )
    # print(rsp)
    return rsp.choices[0].message.content.strip()

# ────────────────────────────────────────────────────────────────
# Code validation – compile‑time
# ────────────────────────────────────────────────────────────────

def _strip_fence(code: str) -> str:
    if code.lstrip().startswith("```"):
        return re.sub(r"^```[a-zA-Z]*|```$", "", code, flags=re.MULTILINE).strip()
    return code.strip()


def _validate_worker(code: str, func_name: str, q):
    """Worker that execs the code, calls func_name(), and sends back either None or an error string."""
    try:
        import cadquery as cq
        ns = {"cq": cq}
        exec(code, ns)
        func = ns.get(func_name)
        if not callable(func):
            q.put(f"{func_name} not defined")
            return
        # run with defaults; if this hangs, the outer process will kill it
        func()
        q.put(None)
    except Exception:
        import traceback as _tb
        q.put(_tb.format_exc())

def validate_code(code: str, func_name: str, timeout: float = 10.0) -> str | None:
    """
    Return None if code.exec() + func() succeeds within `timeout` seconds,
    else return the error message (or 'VALIDATION TIMED OUT').
    """
    from multiprocessing import Process, Queue

    q = Queue()
    p = Process(target=_validate_worker, args=(code, func_name, q), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        # still running → kill it
        p.terminate()
        p.join()
        return "VALIDATION TIMED OUT"

    # get result (should be None or a traceback string)
    try:
        return q.get_nowait()
    except Exception:
        return "VALIDATION FAILED: no result returned"


def refine_code(client, code: str, error: str, abstract: str, detailed: str, func_name: str) -> str:
    user_txt = (
        "The following CadQuery function fails to execute.\n\n" +
        f"### Abstract\n{abstract}\n\n### Detailed\n{detailed}\n\n" +
        "### Current code\n" + code + "\n\n" +
        "### Error\n" + error + "\n\n" +
        "Please reply with **only** the corrected Python function."
    )
    reply = chat(client, [{"role": "user", "content": user_txt}], reasoning_effort="high")
    return _strip_fence(reply)




# ────────────────────────────────────────────────────────────────
# Rendering helpers
# ────────────────────────────────────────────────────────────────

def _shape_to_line_set(shape, resolution):
    # resolution: number of points per edges
    points, lines = list(), list()

    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        curve_adaptor = BRepAdaptor_Curve(edge)
        first_param = curve_adaptor.FirstParameter()
        last_param = curve_adaptor.LastParameter()

        for i in range(resolution):
            param = first_param + (last_param - first_param) * (i / (resolution - 1))
            point = curve_adaptor.Value(param)
            points.append((point.X(), point.Y(), point.Z()))
            if i > 0:
                lines.append((len(points) - 2, len(points) - 1))

        explorer.Next()

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def _create_cylinder(start, end, radius, resolution):
    direction = end - start
    length = np.linalg.norm(direction)

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution)
    cylinder.compute_vertex_normals()

    # Calculate the midpoint and translate the cylinder to this midpoint
    midpoint = (start + end) / 2
    cylinder.translate(midpoint)

    # Rotate the cylinder to align with the line segment direction
    axis = np.array([0, 0, 1])  # The cylinder is initially aligned along the z-axis
    direction_normalized = direction / length  # Normalize the direction vector

    # Calculate the rotation axis and angle to align the cylinder
    rotation_axis = np.cross(axis, direction_normalized)
    if np.linalg.norm(rotation_axis) > 1e-6:  # Check if rotation is necessary
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(axis, direction_normalized), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        cylinder.rotate(R, center=midpoint)

    return cylinder

def _line_set_to_mesh(line_set, radius, resolution):
    mesh = o3d.geometry.TriangleMesh()
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    for line in lines:
        if not np.allclose(points[line[0]], points[line[1]]):
            mesh += _create_cylinder(points[line[0]], points[line[1]], radius, resolution)
    return mesh


def _run_cq_script(code: str):
    ns: Dict[str, Any] = {"cq": cq}
    exec(code, ns)
    res = ns.get("result")
    if res is None:
        raise AttributeError("script does not define 'result'")
    return res.val()


def _cq_script_to_meshes(py_code: str, tmp_dir: str, edge_res=40, tube_res=30, part_name = "part", attempt=0):
    os.makedirs(tmp_dir, exist_ok=True)
    # stem = uuid.uuid4().hex
    step_tmp = Path(tmp_dir) / f"{part_name}_{attempt}.step"
    stl_tmp  = Path(tmp_dir) / f"{part_name}_{attempt}.stl"

    shape = _run_cq_script(py_code)
    cq.exporters.export(shape, str(step_tmp), exportType="STEP")
    cq.exporters.export(shape, str(stl_tmp),  exportType="STL", tolerance=1e-3)

    reader = STEPControl_Reader(); reader.ReadFile(str(step_tmp)); reader.TransferRoots()
    shape_occ = reader.OneShape()
    mesh = o3d.io.read_triangle_mesh(str(stl_tmp))
    lines = _shape_to_line_set(shape_occ, edge_res)
    diag = np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())
    edge_mesh = _line_set_to_mesh(lines, 0.005 * diag, tube_res)
    step_tmp.unlink(missing_ok=True); stl_tmp.unlink(missing_ok=True)


    return mesh, edge_mesh


def _cq_script_to_meshes_process(py_code: str, tmp_dir: str, edge_res=40, tube_res=30, part_name = "part", attempt = 0):
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        # stem = uuid.uuid4().hex
        step_tmp = Path(tmp_dir) / f"{part_name}_{attempt}.step"
        stl_tmp  = Path(tmp_dir) / f"{part_name}_{attempt}.stl"
        edge_tmp = Path(tmp_dir) / f"{part_name}_edges_{attempt}.stl"

        shape = _run_cq_script(py_code)
        cq.exporters.export(shape, str(step_tmp), exportType="STEP")
        cq.exporters.export(shape, str(stl_tmp),  exportType="STL", tolerance=1e-3)

        reader = STEPControl_Reader(); reader.ReadFile(str(step_tmp)); reader.TransferRoots()
        shape_occ = reader.OneShape()
        mesh = o3d.io.read_triangle_mesh(str(stl_tmp))
        lines = _shape_to_line_set(shape_occ, edge_res)
        diag = np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())
        edge_mesh = _line_set_to_mesh(lines, 0.005 * diag, tube_res)
        o3d.io.write_triangle_mesh(str(edge_tmp), edge_mesh)
        step_tmp.unlink(missing_ok=True)
    except:
        pass


    
def _cq_script_to_meshes_safe(
        py_code: str, tmp_dir: str,
        edge_res=40, tube_res=30,
        part_name="part", attempt=0,
        timeout_step: int = TIMEOUT_STEP):

    ctx  = mp.get_context("spawn")           # fresh interpreter
    proc = ctx.Process(
        target=_cq_script_to_meshes_process,
        args=(py_code, tmp_dir, edge_res, tube_res, part_name, attempt)
    )
    proc.start()
    proc.join(timeout_step)
    if proc.is_alive():
        proc.terminate(); proc.join()

    step_tmp = Path(tmp_dir) / f"{part_name}_{attempt}.step"
    stl_tmp  = Path(tmp_dir) / f"{part_name}_{attempt}.stl"
    edge_tmp = Path(tmp_dir) / f"{part_name}_edges_{attempt}.stl"

    # if the child timed-out the files may be missing/empty
    if not stl_tmp.exists() or stl_tmp.stat().st_size == 0:
        step_tmp.unlink(missing_ok=True)
        raise RuntimeError("STL export timed-out or failed.")

    mesh       = o3d.io.read_triangle_mesh(str(stl_tmp))
    edge_mesh  = o3d.io.read_triangle_mesh(str(edge_tmp))
    stl_tmp.unlink(missing_ok=True); edge_tmp.unlink(missing_ok=True)
    return mesh, edge_mesh



    


def _normalise_meshes(m: o3d.geometry.TriangleMesh, e: o3d.geometry.TriangleMesh, rgb, ergb):
    bb = m.get_axis_aligned_bounding_box()
    ctr = bb.get_center()
    m.translate(-ctr)
    e.translate(-ctr)
    ext = max(bb.get_extent())
    if ext > 1e-7:
        m.scale(2.0 / ext, center=np.zeros(3))
        e.scale(2.0 / ext, center=np.zeros(3))
    m.scale(1/2.5, center=np.zeros(3))
    e.scale(1/2.5, center=np.zeros(3))
    m.translate([0.5,0.5,0.5])
    e.translate([0.5,0.5,0.5])
    m.paint_uniform_color(np.asarray(rgb)/255.)
    m.compute_vertex_normals()
    e.paint_uniform_color(np.asarray(ergb)/255.)
    e.compute_vertex_normals()
    return m, e


def _render_view(mesh: o3d.geometry.TriangleMesh,
            edges: o3d.geometry.TriangleMesh,
            front,
            img_res: int,
            width=500, height=500,
            flat_shading=True):
    front = np.asarray(front, dtype=float); front /= np.linalg.norm(front)
    aux = np.array([0., 1., 0.]) if abs(front[0]) < 0.1 and abs(front[1]) < 0.9 else np.array([0., 0., 1.])
    right = np.cross(aux, front); right /= np.linalg.norm(right)+1e-8
    up = np.cross(front, right)

    R = np.column_stack((right, up, front)).T
    eye = np.array([0.5, 0.5, 0.5]) + front * -1.2
    extr = np.eye(4, dtype=float); extr[:3, :3] = R; extr[:3, 3] = -R @ eye

    vis = o3d.visualization.Visualizer(); vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh); vis.add_geometry(edges)
    if flat_shading:
        opt = vis.get_render_option(); opt.light_on = False; opt.background_color = np.ones(3)
    ctrl = vis.get_view_control(); cam = ctrl.convert_to_pinhole_camera_parameters()
    cam.extrinsic = extr; ctrl.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    vis.poll_events(); vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(True))*255; vis.destroy_window()
    img = resize(img.astype(np.uint8), (img_res, img_res), order=2, anti_aliasing=True, preserve_range=True).astype(np.uint8)
    return Image.fromarray(img)




def render_code_to_png(py_script: str,
                              png_path: Path,
                              tmp_dir: str,
                              small: int = 128,
                              pad: int = 16,
                              bg: str = "white",
                              fg: str = "black",
                              part_name: str = "part",
                              attempt:int = 0
                        ):
    solid, edges = _cq_script_to_meshes(py_script, tmp_dir, part_name = part_name, attempt = attempt)
    # solid = _normalise_mesh(solid, (144, 140, 255))
    # edges = _normalise_mesh(edges, (0, 0, 0))
    solid, edges = _normalise_meshes(solid, edges, (144, 140, 255), (0, 0, 0))

    VIEWS = {
        "Isometric": (1, 1, 1),
        "+Y":        (0, 1, 0),
        "-Y":        (0,-1, 0),
        "-Z":        (0, 0,-1),
        "-X":        (-1,0, 0),
        "+Z":        (0, 0, 1),
        "+X":        (1, 0, 0),
    }

    tiles = {}
    for name, front in VIEWS.items():
        res = small*2 if name=="Isometric" else small
        tiles[name] = _render_view(solid, edges, front, res, flat_shading=(name!="Isometric"))

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, 14)
    except (OSError, IOError):
        # fallback if fontconfig can’t find the TTF
        font = ImageFont.load_default()
    lbl_h = font.getbbox("Hg")[3] + 4

    labeled = {}
    for name, im in tiles.items():
        canvas = Image.new("RGB", (im.width, im.height+lbl_h), bg)
        ImageDraw.Draw(canvas).text((4,0), f"View: {name}", font=font, fill=fg)
        canvas.paste(im, (0, lbl_h))
        labeled[name] = canvas

    iso_w, iso_h = labeled["Isometric"].size
    sm_w, sm_h = labeled["+Y"].size
    row1_w = iso_w + pad + sm_w*2 + pad*2
    row2_w = sm_w*4 + pad*3
    total_w, total_h = max(row1_w, row2_w), iso_h + pad + sm_h

    combined = Image.new("RGB", (total_w, total_h), bg)
    x = 0; combined.paste(labeled["Isometric"], (x,0)); x += iso_w+pad
    combined.paste(labeled["+Y"], (x,0)); x += sm_w+pad
    combined.paste(labeled["-Y"], (x,0))
    y = iso_h+pad; x = 0
    for k in ["-Z","-X","+Z","+X"]:
        combined.paste(labeled[k], (x,y)); x += sm_w+pad
    combined = ImageOps.expand(combined, border=3, fill="black")
    combined.save(png_path)


def render_code_to_png_safe(py_script: str,
                              png_path: Path,
                              tmp_dir: str,
                              small: int = 128,
                              pad: int = 16,
                              bg: str = "white",
                              fg: str = "black",
                              part_name: str = "part",
                              attempt = 0,
                              timeout_step = 6*TIMEOUT_STEP):
    # ctx  = mp.get_context("spawn")           # fresh interpreter
    proc = Process(target=render_code_to_png, args=(py_script, png_path, tmp_dir, small, pad, bg, fg, part_name, attempt ))
    proc.daemon = False
    proc.start()
    proc.join(timeout_step)
    if proc.is_alive():
        proc.terminate()
        proc.join()


# ────────────────────────────────────────────────────────────────
# Vision-based validator
# ────────────────────────────────────────────────────────────────
def ask_validator(client, generator_code: str, name: str, description: str,
                  examples: str, error: str | None = None,
                  img_path: Path | None = None) -> str:
    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            sys_msg = (
                "You are an assistant that validates and corrects functions that output a CadQuery shape for given measurements.\n"
                "You will receive **one PNG** that is a montage of **seven views**: an `Isometric` view plus orthographic `+Y`, `-Y`, `-Z`, `-X`, `+Z` and `+X` faces, each labeled accordingly atop the tile.\n"
                # "• The result **must be ONE solid body** (no disjoint solids / loose shells).\n"
                "• The result **must be ONE solid body whose boundary is a single closed, connected shell** (no disjoint solids, loose shells, or internal voids).\n"
                "• You **must** compare that image against the description and ensure **all** described features (holes, fillets, arrays, lofts, etc.) appear in the correct positions, sizes, and counts.\n"
                "• If the shape exactly matches the description, respond with **CORRECT**.\n"
                "• Otherwise respond with **IMPROVE** on the first line, followed by **only** the corrected function body (no explanations).\n\n"
                "Format of the function:\n"
                "  def part_name(param1=p1, param2=p2, ...):\n"
                f"Examples of valid parts you can rely on:\n{examples}"
                f"Examples from the CadQuery website:\n{cq_site_examples}"
            )
            if error:
                user_txt = (
                    f"### Part name\n{name}\n\n### Part description\n{description}"
                    f"\n\n### Function code\n{generator_code}\n\n"
                    f"### Error message\n{error}\n\nPlease fix."
                )
                rsp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role":"system","content":sys_msg},
                              {"role":"user","content":user_txt}]
                )
            else:
                with img_path.open("rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                user_parts = [
                    {"type":"text",
                     "text":("### Part description\n"+description+
                             "\n\n### Function code\n"+generator_code+
                             "\n\nDoes the rendered part match? If not, improve.")},
                    {"type":"image_url",
                     "image_url":{"url":f"data:image/png;base64,{b64}"}}
                ]
                rsp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role":"system","content":sys_msg},
                              {"role":"user","content":user_parts}]
                )
            return rsp.choices[0].message.content.strip()
        except Exception:
            if attempt == MAX_MODEL_RETRIES: raise
            time.sleep(2**attempt)
    raise RuntimeError("all retries exhausted")

# ────────────────────────────────────────────────────────────────
# Shape proposal 
# ────────────────────────────────────────────────────────────────


def propose_shapes(client, db: List[Dict], k: int) -> List[Dict]:
    names = [p["name"] for p in db]
    still_prompt = True
    while still_prompt:
        try:
            if len(names) > 1000:
                use_names = np.random.choice(names, size = 1000)
            else:
                use_names = names
            # print(use_names[:10])
            # existing_names = ", ".join(p["name"] for p in db)
            existing_names = ", ".join(use_names)
            user_prompt = (
                f"We already have these parts: {existing_names}.\n\n"
                f"**Task**: Propose EXACTLY {k} new geometric shapes buildable via boolean ops, sketch edits and rotations, or arrayed copies.\n"
                "⚠️ **Important constraints**:\n"
                "1. Each proposed shape must result in a **single solid body** (no disjoint solids).\n"
                "2. Any pattern/array operation should act only on local features such as holes, teeth, ribs, etc.—**never on the whole part**.\n\n"
                "Your goal is **maximum diversity** across both 2D profiles and 3D solids, with an eye toward **maximal geometrical complexity**:\n"
                "Each proposed shape should be an *incremental* variation of one or more of the existing parts\n"
                "(for instance by tweaking dimensions, adding or removing a feature, or combining features).\n"
                "You don’t have to restrict yourself to purely practical parts, but **feel free to lean on features useful in CAD contexts**—for example:\n"
                "- ribs, mounting tabs or clearances for mechanical assemblies\n"
                "- brackets, gussets or cut-outs for automotive or electronics\n"
                "- lattice, infill patterns or support structures for 3D printing\n"
                "- panels, façade elements or decorative trims for architecture\n"
                "- aerodynamic fairings, control-surface hinges or winglets for aerospace\n"
                "- ergonomic grips, snap-fits or living hinges for consumer products\n"
                "- biomedical-implant surfaces or prosthetic socket interfaces for medical devices\n"
                "- filigrees or prong settings for jewelry design\n"
                "- frames, mounts or end-effectors for robotics\n"
                "- hull segments or bulkheads for shipbuilding\n"
                "- dovetail joints or mortise-tenon connectors for furniture\n"
                "- bosses or inserts for sporting equipment\n\n"
                "**Important**: Name and describe them geometrically (avoid explicit real-world trademarks or part names).\n"
                "Return **ONLY** a JSON array of objects with keys: `name`, `abstract`, `detailed`, `parents`."
            )
            override_system = """
        You are a JSON API.  When the user gives you a JSON schema request, respond with
        *only* valid JSON (no prose, no fences, no extra keys).
        """
            raw = chat(client, [{"role":"system", "content": override_system},{"role": "user", "content": user_prompt}], reasoning_effort=effort)
            raw = _strip_fence(raw)
            props = json.loads(raw)
            seen = {p["name"] for p in db}
            output = [p for p in props if p["name"] not in seen][:k]
            still_prompt = False

        except Exception as e:
            continue
    return output


def generate_code(shape: Dict, db: List[Dict], client) -> Dict:
    """Compile-time → render → vision → geometry gatekeeper loop."""
    abstract, detailed, parents = shape["abstract"], shape["detailed"], shape["parents"]
    
    

    # ---------- 0)  context examples + first draft ---------------------------
    sim = similarity_search(client, db, abstract) 
    if len(parents) > 0:
        par_dicts = [p for p in db if p["name"] in parents]
        sim = sim + par_dicts
    examples = "\n".join(f"# Existing: {p['name']}\n{p['code']}" for p in sim)

    ask = (
        f"Abstract: {abstract}\nDetailed: {detailed}\n\n"
        f"Relevant examples:\n{examples}\n\n"
        "Write a **self-contained** CadQuery v2 function (do NOT call other functions).\n"
        "⚠️ The resulting CadQuery object **must pass a geometry check**:\n"
        "    • exactly one solid body (no disjoint solids)\n"
        "    • watertight; no extra shells\n\n"
        f"def {shape['name']}(…):\n    \"\"\"{abstract} – {detailed}\"\"\"\n    # your code\n"
    )
    code = _strip_fence(chat(client, [{"role": "user", "content": ask}], reasoning_effort=effort))

    # ---------- 1)  unified loop -------------------------------------------
    for attempt in range(1, MAX_REFINES + MAX_VISUAL_REFINES + 1):

        # print(code)

        # 1a · compile-time check
        err = validate_code(code, shape["name"])
        if err:
            code = refine_code(client, code, err, abstract, detailed, shape["name"])
            # print(attempt, code, err)
            continue
        full_source = (
            "import cadquery as cq\nimport math\n"
            + code +
            f"\nresult = {shape['name']}()"
        )
        geo_ok, geo_report = check_geometry({"name": shape["name"], "code": full_source})
        if not geo_ok:
            code = refine_code(client, code,
                               f"Geometry check failed ({geo_report})",
                               abstract, detailed, shape["name"])
            print(shape["name"], geo_report)
            continue

        # 1c · render & vision check
        png_path = TMP_RENDER_DIR / f"{shape['name']}_{attempt}.png"
        
        try:
            full_source = (
                "import cadquery as cq\nimport math\n"
                + code +
                f"\nresult = {shape['name']}()"
            )
            render_code_to_png(full_source, png_path, tmp_dir=str(TMP_RENDER_DIR),
                                    part_name=shape["name"], attempt=attempt)

            reply = ask_validator(
                client, code, shape["name"],
                f"{abstract}\n\n{detailed}", examples=examples,
                img_path=png_path
            )

            if reply.startswith("IMPROVE"):
                code = _strip_fence("\n".join(reply.splitlines()[1:]))
                continue
            if not reply.startswith("CORRECT"):
                raise RuntimeError(f"Validator returned: {reply!r}")

        except Exception as render_exc:
            code = refine_code(client, code, traceback.format_exc(),
                               abstract, detailed, shape["name"])
            print("1", render_exc)
            continue

        

        # All gates passed 🎉
        break
    else:
        raise RuntimeError(f"{shape['name']} failed after {attempt} attempts.")

    # ---------- 2)  embed + return -----------------------------------------
    shape["code"] = code
    emb = get_embedding(client, abstract)
    shape["embedding_path"] = save_embedding(shape["name"], emb)
    return shape

def _generate_single(prop, db_snapshot):
    """Executed in the worker. We *must* re-create anything that
    is not fork-safe (OpenAI client, temp dirs, etc.)."""
    # 1.  rebuild OpenAI client *inside* the worker
    from openai import OpenAI
    _client = OpenAI(api_key=API_KEY)
    try:
        # 2.  call your existing helper (it only needs read-only DB)
        part = generate_code(prop, db_snapshot, _client)
        return part, None
    except Exception as e:
        return prop, e

       


def generate_level(client, db: list[dict], lvl: int):
    print(
        f"\n=== Level {lvl}: generating {SHAPES_PER_LEVEL} shapes "
        f"with {N_WORKERS} workers ==="
    )

    proposals   = propose_shapes(client, db, SHAPES_PER_LEVEL)
    db_snapshot = tuple(db)                     # immutable copy

    parts_ok, parts_failed = [], []
    with NonDaemonPool(N_WORKERS) as pool:
        for fut in tqdm(
            pool.imap_unordered(
                partial(
                    _generate_single,
                    db_snapshot=db_snapshot
                ),
                proposals,
                chunksize=1
            ), total=len(proposals)
        ):
            part, exc = fut
            if exc is None:
                parts_ok.append(part)
                print("✔", part["name"])
            else:
                print("⚠", part["name"], "-", exc)

    db.extend(parts_ok)
    save_db(db)







if __name__ == "__main__":
    from openai import OpenAI
    db = load_db()
    seed_primitives(client,db)
    save_db(db)
    for lvl in range(1, MAX_LEVELS + 1):
        generate_level(client, db, lvl)
    print("\n🎉 Generation complete. Total parts in DB:", len(db))



