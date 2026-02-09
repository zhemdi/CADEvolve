from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopoDS import TopoDS_Solid, TopoDS
from OCP.TopAbs import TopAbs_Orientation, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
import os
import resource
import numpy as np
import cadquery as cq

# =========== Pooling ==========

class Wrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, args, shared_args_idx):
        while shared_args_idx.value < len(args):
            i = shared_args_idx.value
            item = args[i]
            try:
                if isinstance(item, dict):
                    self.func(item)
            except Exception:
                pass
            finally:
                shared_args_idx.value += 1

class ProcessPool:
    def __init__(self, task_func, task_args: list[tuple], n_processes: int = 16, timeout: float = 30):
        self.n_processes = n_processes
        self.timeout = timeout
        self.task_func = task_func
        self.task_args = task_args

    def run(self):
        import time
        from multiprocessing import Process, Value
        from tqdm import tqdm

        pbar = tqdm(total=len(self.task_args))
        task_args = self.split_list(self.task_args, self.n_processes)
        shared_args_indicies = [Value('i', 0) for _ in range(self.n_processes)]
        last_args_indicies = [0 for _ in range(self.n_processes)]
        pool = [Process(target=Wrapper(self.task_func), args=(task_args[i], shared_args_indicies[i]), daemon=True)
                for i in range(self.n_processes)]
        unprocessed_args = []

        for process in pool:
            process.start()

        n_processes_running = self.n_processes

        while n_processes_running:
            time.sleep(self.timeout)
            for i in range(self.n_processes):
                if pool[i] is None:
                    continue

                if shared_args_indicies[i].value >= len(task_args[i]):
                    process = pool[i]
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=self.timeout)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    pool[i] = None
                    n_processes_running -= 1
                elif shared_args_indicies[i].value == last_args_indicies[i]:
                    hang_process = pool[i]
                    hang_process.terminate()
                    hang_process.join(timeout=self.timeout)
                    if hang_process.is_alive():
                        hang_process.kill()
                        hang_process.join()

                    unprocessed_args.append(task_args[i][shared_args_indicies[i].value])
                    # with shared_args_indicies[i].get_lock():
                    shared_args_indicies[i].value += 1
                    new_process = Process(
                        target=Wrapper(self.task_func), args=(task_args[i], shared_args_indicies[i]), daemon=True
                    )
                    new_process.start()
                    pool[i] = new_process

                last_args_indicies[i] = shared_args_indicies[i].value

            pbar.update(sum(last_args_indicies) - pbar.n)

        pbar.close()

        return unprocessed_args
    
    def split_list(self, lst, num_chunks):
        n = len(lst)
        base_size, remainder = divmod(n, num_chunks)
        sizes = [base_size + (1 if i < remainder else 0) for i in range(num_chunks)]

        chunks = []
        start = 0
        for size in sizes:
            chunks.append(lst[start: start + size])
            start += size
        return chunks

# =========== Write log ==========

def log_line(log_path: str, detail: str, msg: str, script: str | None = None):
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            key = f"{detail}/{script}" if script else detail
            os.write(fd, (f"{key}: {msg}\n").encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass

# =========== Save the code ==========

def save_code(path: str, code_str: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code_str)


# =========== Set memory threshold (if applicable) ==========

def set_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def compute_mem_per_worker(system_reserved_ratio=0.2, n_workers=32) -> int:
    total_mem_bytes = None

    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            s = f.read().strip()
        if s != "max":
            total_mem_bytes = int(s)
    except Exception:
        total_mem_bytes = None

    return int(total_mem_bytes * (1 - system_reserved_ratio) / max(1, int(n_workers)))


def apply_as_limit(bytes_soft: int):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = int(bytes_soft)
    if hard != resource.RLIM_INFINITY:
        new_soft = min(new_soft, hard)

    resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))

# =========== Find the center of CQ BoundingBox ==========

def find_center(bbox, eps: float = 1e-7):
    
    c = ((bbox.xmin + bbox.xmax) / 2,
                (bbox.ymin + bbox.ymax) / 2,
                (bbox.zmin + bbox.zmax) / 2)

    return tuple(0.0 if abs(v) < eps else v for v in c)

# =========== Tesselation of the solid ==========

def tessellate_solid(solid: TopoDS_Solid, tolerance: float = 0.1, angular_tolerance: float = 0.1):
    mesh = BRepMesh_IncrementalMesh(solid, tolerance, True, angular_tolerance)
    mesh.Perform()
    
    vertices = []
    triangles = []
    vertex_offset = 0 

    explorer = TopExp_Explorer(solid, TopAbs_FACE)

    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face, loc)

        if poly is None:
            explorer.Next()
            continue

        trsf = loc.Transformation()
        is_reversed = (face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED)

        face_vertices = []
        for i in range(1, poly.NbNodes() + 1):
            p = poly.Node(i).Transformed(trsf)
            face_vertices.append([p.X(), p.Y(), p.Z()])
        
        face_triangles = []
        for i in range(1, poly.NbTriangles() + 1): 
            tri = poly.Triangle(i)
            i1, i2, i3 = tri.Value(1), tri.Value(2), tri.Value(3)
            if is_reversed:
                face_triangles.append([i1 - 1 + vertex_offset, i3 - 1 + vertex_offset, i2 - 1 + vertex_offset])
            else:
                face_triangles.append([i1 - 1 + vertex_offset, i2 - 1 + vertex_offset, i3 - 1 + vertex_offset])
        
        vertices.extend(face_vertices)
        triangles.extend(face_triangles)
        vertex_offset += len(face_vertices)
        explorer.Next()

    vertices = np.array(vertices)
    triangles = np.array(triangles)
        
    return vertices, triangles

# =========== Validate script ========== 

def extract_subsolids(topo_shape):
    solids = []
    exp = TopExp_Explorer(topo_shape, TopAbs_SOLID)
    while exp.More():
        solids.append(exp.Current())
        exp.Next()
    return solids

def count_subshapes(topo_shape, shape_type):
    explorer = TopExp_Explorer(topo_shape, shape_type)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count

def min_distance_between_shapes(s1, s2) -> float:
    dist_op = BRepExtrema_DistShapeShape(s1, s2)
    dist_op.Perform()
    if not dist_op.IsDone():
        return float("inf")
    return dist_op.Value()

def check_shape_validity(topo_shape):
    try:
        top_solids = extract_subsolids(topo_shape)
        if len(top_solids) == 0:
            c1 = False
        elif len(top_solids) == 1:
            c1 = True
        else:
            bbox = cq.Shape.cast(topo_shape).BoundingBox()
            tol = max(bbox.xlen, bbox.ylen, bbox.zlen) * 0.01
            all_connected = True
            for i in range(len(top_solids)):
                connected_i = False
                for j in range(len(top_solids)):
                    if i == j:
                        continue
                    d = min_distance_between_shapes(top_solids[i], top_solids[j])
                    if d < tol:
                        connected_i = True
                        break
                if not connected_i:
                    all_connected = False
                    break
            c1 = all_connected
        top_type = topo_shape.ShapeType()
        c2 = (top_type == TopAbs_SOLID or top_type == TopAbs_COMPOUND)
        shells_count = count_subshapes(topo_shape, TopAbs_SHELL)
        c3 = (shells_count == len(top_solids) and c1)
    except Exception:
        return False, False, False
    return c1, c2, c3
