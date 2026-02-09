import os, ast, argparse, json, tempfile, traceback
import cadquery as cq
import numpy as np
from pathlib import Path
from cadquery import Plane as _CQPlane, Vector as _CQVector
from cadquery.selectors import Selector as _CQSelectorBase
from utils.canonicalization.helpers import tessellate_solid, log_line, check_shape_validity, save_code, ProcessPool

LOG = []
_NEXT_ID = 1
_ORIG = {}
_PATCHED = False
_CALL_DEPTH = 0
_MISSING = object()
SELECTOR_WP_METHODS = {"solids", "faces", "edges", "wires", "shells", "vertices"}


def save_json(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ops": LOG}, f, ensure_ascii=False, indent=2)


def _is_selector_obj(x) -> bool:
    if _CQSelectorBase is None:
        return False
    try:
        return isinstance(x, _CQSelectorBase)
    except Exception:
        return False

def _new_id():
    global _NEXT_ID
    i = _NEXT_ID; _NEXT_ID += 1; return i


def _tag(obj, op_id):
    if isinstance(obj, cq.Workplane):
        try: setattr(obj, "_cq_wp_id", op_id)
        except Exception: pass
    elif isinstance(obj, cq.Shape):
        try: setattr(obj, "_cq_shape_id", op_id)
        except Exception: pass
    elif _is_selector_obj(obj):
        try: setattr(obj, "_cq_sel_id", op_id)
        except Exception:
            try: object.__setattr__(obj, "_cq_sel_id", op_id)
            except Exception: pass

    return obj


def _get_id(obj):
    if isinstance(obj, cq.Workplane):
        return getattr(obj, "_cq_wp_id", None)
    elif isinstance(obj, cq.Shape):
        return getattr(obj, "_cq_shape_id", None)
    elif _is_selector_obj(obj):
        return getattr(obj, "_cq_sel_id", None)


def _ser(v):
    if isinstance(v, cq.Workplane):
        return {"__wp_ref__": _get_id(v)}
    if isinstance(v, cq.Shape):
        return {"__shape_ref__": _get_id(v)}
    if _is_selector_obj(v):
        sid = _get_id(v)
        return {"__sel_ref__": sid}
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        if len(v) > 0 and all(isinstance(x, cq.Shape) for x in v):
            return [{"__shape_ref__": _get_id(x)} for x in v]
        return [_ser(x) for x in v]
    if isinstance(v, dict):
        return {k: _ser(x) for k, x in v.items()}
    if isinstance(v, _CQVector):
        return {"__type__": "Vector", "value": [float(v.x), float(v.y), float(v.z)]}
    if isinstance(v, _CQPlane):
        normal = getattr(v, "normal", getattr(v, "zDir", None))
        return {
            "__type__": "Plane",
            "origin": _ser(v.origin),
            "xDir":   _ser(v.xDir),
            "normal": _ser(normal) if normal is not None else None,
        }
    return repr(v)


def patch():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # Deepcopy and Copy
    _ORIG[(cq.Shape, "__deepcopy__")] = getattr(cq.Shape, "__deepcopy__", _MISSING)
    _ORIG[(cq.Shape, "__copy__")]     = getattr(cq.Shape, "__copy__", _MISSING)

    def _shape_copy(self):
        return self

    def _shape_deepcopy(self, memo):
        return self

    setattr(cq.Shape, "__copy__", _shape_copy)
    setattr(cq.Shape, "__deepcopy__", _shape_deepcopy)

    # 1) Workplane.__init__
    _ORIG[(cq.Workplane, "__init__")] = cq.Workplane.__init__

    def _init_hook(self, *args, **kwargs):
        global _CALL_DEPTH
        is_top = (_CALL_DEPTH == 0)

        _CALL_DEPTH += 1
        try:
            _ORIG[(cq.Workplane, "__init__")](self, *args, **kwargs)
        finally:
            _CALL_DEPTH -= 1

        if not is_top:
            return

        in_plane = kwargs.get("inPlane", None)
        if in_plane is None and len(args) >= 1:
            in_plane = args[0]

        if in_plane is None:
            in_plane = 'XY'


        obj_val = kwargs.get("obj", None)
        if obj_val is None and len(args) >= 2:
            obj_val = args[1]

        if obj_val is not None and not isinstance(obj_val, (str, _CQPlane)):
            return
        if not isinstance(in_plane, (str, _CQPlane)):
            return

        op_id = _new_id()
        LOG.append({
            "id": op_id,
            "op": "Workplane",
            "args": [_ser(in_plane)],
            "kwargs": {}
        })
        _tag(self, op_id)
        _bind_existing_attrs(op_id, self)

    cq.Workplane.__init__ = _init_hook

    # 1.1) Workplane.__setattr__
    _ORIG[(cq.Workplane, "__setattr__")] = getattr(cq.Workplane, "__setattr__", object.__setattr__)
    def _wp_setattr(self, name, value):
        global _CALL_DEPTH
        is_top = (_CALL_DEPTH == 0)
        base_id = _get_id(self)
        if is_top and base_id is not None and not str(name).startswith("_cq_"):
            op_id = _new_id()
            LOG.append({
                "id": op_id,
                "op": "setattr",
                "on": base_id,
                "obj_type": "attr",
                "attr": str(name),
                "value": _ser(value),
            })
        ret = _ORIG[(cq.Workplane, "__setattr__")](self, name, value)
        if base_id is not None and not str(name).startswith("_cq_"):
            _bind_obj(base_id, value, str(name))
        return ret
    cq.Workplane.__setattr__ = _wp_setattr

    # 1.2) Shape.__setattr__
    _ORIG[(cq.Shape, "__setattr__")] = getattr(cq.Shape, "__setattr__", object.__setattr__)
    def _shape_setattr(self, name, value):
        global _CALL_DEPTH
        is_top = (_CALL_DEPTH == 0)
        base_id = _get_id(self)
        if is_top and base_id is not None and not str(name).startswith("_cq_"):
            op_id = _new_id()
            LOG.append({
                "id": op_id,
                "op": "setattr",
                "on": base_id,
                "obj_type": "attr",
                "attr": str(name),
                "value": _ser(value),
            })
        ret = _ORIG[(cq.Shape, "__setattr__")](self, name, value)
        if base_id is not None and not str(name).startswith("_cq_"):
            _bind_obj(base_id, value, str(name))
        return ret
    cq.Shape.__setattr__ = _shape_setattr

    # 2) universal wrapper for all public methods of cq.Workplane
    for name in dir(cq.Workplane):
        if name.startswith("_"):
            continue
        if name in ("toSvg", "toSvgString"):
            continue
        attr = getattr(cq.Workplane, name)
        if not callable(attr):
            continue

        _ORIG[(cq.Workplane, name)] = attr

        def make_wp_wrapper(method_name, orig):
            def wrapper(self, *args, __orig=orig, __name=method_name, **kwargs):
                global _CALL_DEPTH
                is_top = (_CALL_DEPTH == 0)

                is_selector = (__name in SELECTOR_WP_METHODS)

                _CALL_DEPTH += 1
                try:
                    res = __orig(self, *args, **kwargs)
                finally:
                    _CALL_DEPTH -= 1

                if is_top:
                    if __name == "all":
                        base_id = _get_id(self)
                        try:
                            items = list(res)
                        except TypeError:
                            items = []

                        op_id = _new_id()
                        shapes_meta = []
                        for idx, itm in enumerate(items):
                            if isinstance(itm, cq.Shape) or hasattr(itm, "wrapped"):
                                sid = _new_id()
                                _tag(itm, sid)
                                shapes_meta.append({"sid": sid, "index": idx, "kind": "shape"})
                            elif isinstance(itm, cq.Workplane):
                                wid = _new_id()
                                _tag(itm, wid)
                                shapes_meta.append({"sid": wid, "index": idx, "kind": "wp"})

                        LOG.append({
                            "id": op_id,
                            "op": __name,
                            "on": base_id,
                            "obj_type": "shape_list",
                            "shapes": shapes_meta,
                            "args": [_ser(a) for a in args],
                            "kwargs": {k: _ser(v) for k, v in kwargs.items()},
                        })
                        return items

                    elif isinstance(res, cq.Workplane):
                        base_id = _get_id(self)
                        op_id = _new_id()
                        LOG.append({
                            "id": op_id,
                            "op": __name,
                            "on": base_id,
                            "obj_type": "wp",
                            "args": [_ser(a) for a in args],
                            "kwargs": {k: _ser(v) for k, v in kwargs.items()},
                            "is_selector": is_selector
                        })
                        _tag(res, op_id)
                        _bind_existing_attrs(op_id, res)

                        if __name in SELECTOR_WP_METHODS:
                            _CALL_DEPTH += 1
                            try:
                                try:
                                    seq = list(res.vals())
                                except Exception:
                                    seq = []
                            finally:
                                _CALL_DEPTH -= 1

                            shapes_meta = []
                            for idx, shp in enumerate(seq):
                                if isinstance(shp, cq.Shape) or hasattr(shp, "wrapped"):
                                    sid = _get_id(shp)
                                    if sid is None:
                                        sid = _new_id()
                                        _tag(shp, sid)
                                    shapes_meta.append({"sid": sid, "index": idx})

                            if shapes_meta:
                                LOG.append({
                                    "id": _new_id(),
                                    "op": "__vals__",
                                    "on_wp": op_id,
                                    "obj_type": "shape_from_wp",
                                    "shapes": shapes_meta
                                })

                    elif isinstance(res, cq.Shape):
                        base_id = _get_id(self)
                        op_id = _new_id()
                        LOG.append({
                            "id": op_id,
                            "op": __name,
                            "on": base_id,
                            "obj_type": "shape",
                            "args": [_ser(a) for a in args],
                            "kwargs": {k: _ser(v) for k, v in kwargs.items()}
                        })
                        _tag(res, op_id)

                    elif isinstance(res, (list, tuple)) and res and all(isinstance(x, cq.Shape) for x in res):
                        base_id = _get_id(self)
                        op_id = _new_id()
                        shapes_meta = []
                        for idx, shp in enumerate(res):
                            sid = _new_id()
                            _tag(shp, sid)
                            shapes_meta.append({"sid": sid, "index": idx})
                        LOG.append({
                            "id": op_id,
                            "op": __name,
                            "on": base_id,
                            "obj_type": "shape_list",
                            "shapes": shapes_meta,
                            "args": [_ser(a) for a in args],
                            "kwargs": {k: _ser(v) for k, v in kwargs.items()}
                        })
                return res
            return wrapper

        setattr(cq.Workplane, name, make_wp_wrapper(name, attr))

    # 3) universal wrapper for all public methods of cq.Shape
    AGGREGATORS = ("Solids", "Faces", "Edges", "Wires", "Shells", "Vertices", "Compounds")
    for name in AGGREGATORS:
        attr = getattr(cq.Shape, name, None)
        if not callable(attr):
            continue

        _ORIG[(cq.Shape, name)] = attr

        def make_shape_wrapper(method_name, orig):
            def wrapper(self, *args, __orig=orig, __name=method_name, **kwargs):
                global _CALL_DEPTH
                is_top = (_CALL_DEPTH == 0)

                _CALL_DEPTH += 1
                try:
                    res = __orig(self, *args, **kwargs)
                finally:
                    _CALL_DEPTH -= 1

                if is_top:
                    base_id = _get_id(self)
                    op_id = _new_id()
                    try:
                        items = list(res) if res is not None else []
                    except TypeError:
                        items = []

                    shapes_meta = []
                    for idx, shp in enumerate(items):
                        if isinstance(shp, cq.Shape) or hasattr(shp, "wrapped"):
                            sid = _new_id()
                            _tag(shp, sid)
                            shapes_meta.append({"sid": sid, "index": idx})

                    LOG.append({
                        "id": op_id,
                        "op": __name,
                        "on": base_id,
                        "obj_type": "shape_list",
                        "shapes": shapes_meta,
                        "args": [_ser(a) for a in args],
                        "kwargs": {k: _ser(v) for k, v in kwargs.items()}
                    })
                return res
            return wrapper

        setattr(cq.Shape, name, make_shape_wrapper(name, attr))
    
    # 4) wrapper for cq.NearestToPointSelector
    Sel = getattr(cq, "NearestToPointSelector", None)
    if Sel is not None and hasattr(Sel, "__init__"):
        _ORIG[(Sel, "__init__")] = Sel.__init__

        def _sel_init_hook(self, *args, **kwargs):
            global _CALL_DEPTH
            is_top = (_CALL_DEPTH == 0)

            _CALL_DEPTH += 1
            try:
                _ORIG[(Sel, "__init__")](self, *args, **kwargs)
            finally:
                _CALL_DEPTH -= 1

            if not is_top:
                return

            op_id = _new_id()
            LOG.append({
                "id": op_id,
                "op": "NearestToPointSelector",
                "on": None,
                "obj_type": "selector",
                "args": [_ser(a) for a in args],
                "kwargs": {k: _ser(v) for k, v in kwargs.items()},
            })
            _tag(self, op_id)

        Sel.__init__ = _sel_init_hook

    def _wrap_shape_factories():
        SHAPE_CLS = (cq.Edge, cq.Wire, cq.Face, cq.Shell, cq.Solid, cq.Compound, cq.Vertex)


        for cls in SHAPE_CLS:
            for name in dir(cls):
                if name.startswith("_"):
                    continue

                if name in ("Solids", "Faces", "Edges", "Wires", "Shells", "Vertices", "Compounds"):
                    continue

                attr = getattr(cls, name)

                if not callable(attr):
                    continue

                _ORIG[(cls, name)] = attr

                def make_factory_wrapper(__cls=cls, __name=name, __orig=attr):
                    def wrapper(*args, **kwargs):
                        global _CALL_DEPTH
                        is_top = (_CALL_DEPTH == 0)

                        _CALL_DEPTH += 1
                        try:
                            res = __orig(*args, **kwargs)
                        finally:
                            _CALL_DEPTH -= 1

                        if is_top and isinstance(res, cq.Shape):
                            op_id = _new_id()
                            LOG.append({
                                "id": op_id,
                                "op": __name,
                                "cls": __cls.__name__,
                                "on": None,
                                "obj_type": "shape",
                                "args": [_ser(a) for a in args],
                                "kwargs": {k: _ser(v) for k, v in kwargs.items()},
                            })
                            _tag(res, op_id)
                        return res
                    return wrapper

                setattr(cls, name, make_factory_wrapper())

    _wrap_shape_factories()


def _patch_object_class(obj):
    if obj is None:
        return
    cls = type(obj)

    if cls in (int, float, bool, str, bytes, bytearray,
               list, tuple, dict, set, frozenset, range, memoryview, complex, type(None)):
        return

    k = (cls, "__setattr__")
    if k in _ORIG:
        return

    prev = getattr(cls, "__setattr__", object.__setattr__)

    def _obj_setattr(self, name, value, __prev=prev):
        global _CALL_DEPTH
        is_top = (_CALL_DEPTH == 0)
        owner_id = getattr(self, "_cq_owner_id", None)
        path = getattr(self, "_cq_attr_path", None)
        if is_top and owner_id is not None and not str(name).startswith("_cq_"):
            op_id = _new_id()
            attr_path = f"{path}.{name}" if path else str(name)
            LOG.append({
                "id": op_id,
                "op": "setattr",
                "on": owner_id,
                "obj_type": "attr",
                "attr": attr_path,
                "value": _ser(value),
            })
        ret = __prev(self, name, value)
        if owner_id is not None and not str(name).startswith("_cq_"):
            _bind_obj(owner_id, value, f"{path}.{name}" if path else str(name))
        return ret

    try:
        setattr(cls, "__setattr__", _obj_setattr)
    except (TypeError, AttributeError):
        return

    _ORIG[k] = prev


def _bind_obj(owner_id, obj, path):
    if obj is None:
        return
    try:
        object.__setattr__(obj, "_cq_owner_id", owner_id)
        object.__setattr__(obj, "_cq_attr_path", str(path))
    except Exception:
        pass
    _patch_object_class(obj)


def _bind_existing_attrs(owner_id, obj):
    try:
        d = getattr(obj, "__dict__", None)
        if not d:
            return
        for n, v in d.items():
            if str(n).startswith("_cq_"):
                continue
            _bind_obj(owner_id, v, n)
    except Exception:
        pass


def unpatch():
    global _PATCHED
    if not _PATCHED:
        return
    for (cls, name), orig in list(_ORIG.items()):
        try:
            if orig is _MISSING:
                try:
                    delattr(cls, name)
                except Exception:
                    pass
            else:
                setattr(cls, name, orig)
        except Exception:
            pass
    _ORIG.clear()
    _PATCHED = False


def _collect_needed_shape_ids(log):
    needed = set()
    def visit(x):
        if isinstance(x, dict):
            if "__shape_ref__" in x and x["__shape_ref__"] is not None:
                needed.add(x["__shape_ref__"])
            else:
                for v in x.values():
                    visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
    for op in log:
        visit(op.get("args", []))
        visit(op.get("kwargs", {}))
    return needed


def _py_map(v, idmap):
    if isinstance(v, (bool, type(None), str, int, float)):
        return repr(v)
    if isinstance(v, dict):
        if "__wp_ref__" in v:
            oid = v["__wp_ref__"]
            return f"wp{idmap.get(oid, oid)}"
        if "__shape_ref__" in v:
            sid = v["__shape_ref__"]
            return f"wp{idmap.get(sid, sid)}"
        if "__sel_ref__" in v:
            sel_id = v["__sel_ref__"]
            return f"wp{idmap.get(sel_id, sel_id)}"
        if v.get('__type__') == 'Plane':
            def _v(dd):
                vv = dd["value"]; return f"{repr(vv[0])}, {repr(vv[1])}, {repr(vv[2])}"
            o = v["origin"]; xd = v["xDir"]; n = v.get("normal")
            normal_expr = f"cq.Vector({_v(n)})" if n is not None else "None"
            return ("cq.Plane("
                    f"origin=cq.Vector({_v(o)}), "
                    f"xDir=cq.Vector({_v(xd)}), "
                    f"normal={normal_expr})")
        if v.get('__type__') == "Vector":
            x, y, z = v.get('value')
            return f"cq.Vector({repr(x)}, {repr(y)}, {repr(z)})"
        return repr({k: v[k] for k in v})
    if isinstance(v, list):
        return "[" + ", ".join(_py_map(x, idmap) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(_py_map(x, idmap) for x in v) + ")"
    return repr(v)


def standardize_code(log, final_var="result"):
    lines = ["import cadquery as cq"]

    needed_shapes = _collect_needed_shape_ids(log)

    idmap = {}
    next_id = 1
    def mid(x):
        nonlocal next_id
        if x not in idmap:
            idmap[x] = next_id
            next_id += 1
        return idmap[x]

    last_wp_mapped = None

    for op in log:
        oid = op["id"]
        obj_type = op.get("obj_type", "wp")

        if op.get("op") == "__vals__":
            wp_id = op["on_wp"]
            for item in sorted(op.get("shapes", []), key=lambda t: t["index"]):
                sid = item["sid"]; idx = item["index"]
                if sid in needed_shapes:
                    lines.append(f"wp{mid(sid)} = wp{mid(wp_id)}.vals()[{idx}]")
            continue

        if obj_type == "selector":
            args = op.get("args", [])
            kwargs = op.get("kwargs", {})
            parts = []
            if args:
                parts += [_py_map(a, idmap) for a in args]
            if kwargs:
                parts += [f"{k}={_py_map(v, idmap)}" for k, v in sorted(kwargs.items())]
            call = ", ".join(parts)
            lines.append(f"wp{mid(oid)} = cq.NearestToPointSelector({call})")
            continue

        if op["op"] == "Workplane":
            arg = _py_map(op["args"][0], idmap) if op.get("args") else ""
            lines.append(f"wp{mid(oid)} = cq.Workplane({arg})")
            last_wp_mapped = mid(oid)
            continue

        if op["op"] == "setattr":
            on = op["on"]; attr = op["attr"]; val = _py_map(op.get("value"), idmap)
            lines.append(f"wp{mid(on)}.{attr} = {val}")
            continue

        on = op.get("on")
        args = op.get("args", [])
        kwargs = op.get("kwargs", {})
        parts = []
        if args:   parts += [_py_map(a, idmap) for a in args]
        if kwargs: parts += [f"{k}={_py_map(v, idmap)}" for k, v in sorted(kwargs.items())]
        call = ", ".join(parts)

        if obj_type == "shape":
            if on is None and "cls" in op:
                cls_name = op["cls"]
                lines.append(f"wp{mid(oid)} = cq.{cls_name}.{op['op']}({call})")
            else:
                lines.append(f"wp{mid(oid)} = wp{mid(on)}.{op['op']}({call})")
            continue

        if obj_type == "shape_list":
            lines.append(f"wp{mid(oid)} = list(wp{mid(on)}.{op['op']}({call}))")
            for item in sorted(op.get("shapes", []), key=lambda t: t["index"]):
                sid = item["sid"]; idx = item["index"]
                if sid in needed_shapes:
                    lines.append(f"wp{mid(sid)} = wp{mid(oid)}[{idx}]")
            continue

        if obj_type == "wp":
            lines.append(f"wp{mid(oid)} = wp{mid(on)}.{op['op']}({call})")
            if not op.get("is_selector", False):
                last_wp_mapped = mid(oid)
            continue

    lines.append(f"{final_var} = wp{last_wp_mapped}" if last_wp_mapped is not None else f"{final_var} = None")
    return "\n".join(lines)


class ParametricCurveTransformer(ast.NodeTransformer):

    def __init__(self, num_points=25):
        super().__init__()
        self.num_points = num_points

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        func_attr = node.func
        if isinstance(func_attr, ast.Attribute) and func_attr.attr == "parametricCurve":
            return self._rewrite_parametric_curve(node, func_attr.value)

        return node

    def _rewrite_parametric_curve(self, node: ast.Call, base: ast.AST):

        param_order = [
            "func",
            "N",
            "start",
            "stop",
            "tol",
            "minDeg",
            "maxDeg",
            "smoothing",
            "makeWire",
        ]

        values = {name: None for name in param_order}

        for i, arg in enumerate(node.args):
            if i >= len(param_order):
                break
            values[param_order[i]] = arg

        for kw in node.keywords:
            if kw.arg in values:
                values[kw.arg] = kw.value

        callback = values["func"]

        N = ast.Constant(value=self.num_points)

        start = values["start"] if values["start"] is not None else ast.Constant(0.0)
        stop = values["stop"] if values["stop"] is not None else ast.Constant(1.0)

        tol = values["tol"]
        min_deg = values["minDeg"]
        max_deg = values["maxDeg"]
        smoothing = values["smoothing"]

        make_wire = (
            values["makeWire"] if values["makeWire"] is not None else ast.Constant(True)
        )

        t_store = ast.Name(id="t", ctx=ast.Store())
        t_load = ast.Name(id="t", ctx=ast.Load())

        diff = ast.BinOp(left=stop, op=ast.Sub(), right=start)
        t_over_N = ast.BinOp(left=t_load, op=ast.Div(), right=N)
        mul = ast.BinOp(left=diff, op=ast.Mult(), right=t_over_N)
        param = ast.BinOp(left=start, op=ast.Add(), right=mul)

        func_call = ast.Call(func=callback, args=[param], keywords=[])

        range_call = ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[ast.BinOp(left=N, op=ast.Add(), right=ast.Constant(1))],
            keywords=[],
        )

        comp = ast.comprehension(
            target=t_store,
            iter=range_call,
            ifs=[],
            is_async=0,
        )

        gen = ast.GeneratorExp(elt=func_call, generators=[comp])

        new_keywords = []
        if tol is not None:
            new_keywords.append(ast.keyword(arg="tol", value=tol))
        if min_deg is not None:
            new_keywords.append(ast.keyword(arg="minDeg", value=min_deg))
        if max_deg is not None:
            new_keywords.append(ast.keyword(arg="maxDeg", value=max_deg))
        if smoothing is not None:
            new_keywords.append(ast.keyword(arg="smoothing", value=smoothing))

        new_keywords.append(ast.keyword(arg="makeWire", value=make_wire))

        points_list = ast.Call(
            func=ast.Name(id="list", ctx=ast.Load()),
            args=[gen],
            keywords=[],
        )
        
        new_call = ast.Call(
            func=ast.Attribute(value=base, attr="splineApprox", ctx=ast.Load()),
            args=[points_list],
            keywords=new_keywords,
        )

        return new_call


def replace_parametric_curve(source: str, num_points) -> str:
    tree = ast.parse(source)
    transformer = ParametricCurveTransformer(num_points=num_points)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


class ParametricLambdaToFunc(ast.NodeTransformer):

    def __init__(self, base_name: str = "func"):
        super().__init__()
        self.base_name = base_name
        self.func_counter = 0
        self.func_defs: list[ast.FunctionDef] = []

    def _make_func(self, lambda_node: ast.Lambda) -> ast.Name:
        name = self.base_name if self.func_counter == 0 else f"{self.base_name}_{self.func_counter}"
        self.func_counter += 1

        func_def = ast.FunctionDef(
            name=name,
            args=lambda_node.args,
            body=[ast.Return(value=lambda_node.body)],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        self.func_defs.append(func_def)
        return ast.Name(id=name, ctx=ast.Load())

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "parametricCurve":
            new_args = []
            for arg in node.args:
                if isinstance(arg, ast.Lambda):
                    new_args.append(self._make_func(arg))
                else:
                    new_args.append(arg)
            node.args = new_args
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node = self.generic_visit(node)

        imports = []
        others = []
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.append(stmt)
            else:
                others.append(stmt)

        node.body = imports + self.func_defs + others
        return node


def del_lambda(code: str, base_name: str = "func") -> str:
    tree = ast.parse(code)
    transformer = ParametricLambdaToFunc(base_name=base_name)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def process_detail(task):
    """
    task: {
        "detail": str,
        "script_path": str,
        "standardized_dir": str,
        "logs_path": str,
        "flag_parametricCurve": bool
    }
    """
    detail_name = task["detail"]
    script_path = Path(task["script_path"])
    standardized_dir = Path(task["standardized_dir"])
    logs_path = task["logs_path"]
    flag_parametricCurve = task["flag_parametricCurve"]

    is_flat = (detail_name == script_path.stem)
    script_id = None if is_flat else script_path.stem

    rec = {
        "detail": detail_name,
        "script": script_path.name,
        "flag_parametricCurve": flag_parametricCurve,
        "stages": {},
        "attempts": [],
        "is_different": None,
        "success": False,
        "error": None,
    }

    global _NEXT_ID, _CALL_DEPTH
    _NEXT_ID = 1
    _CALL_DEPTH = 0

    def set_error(stage: str, e: Exception | None = None, tb: str | None = None, msg: str | None = None):
        if rec["error"] is None:
            rec["error"] = {
                "stage": stage,
                "message": (msg if msg is not None else (str(e) if e is not None else "")),
                "traceback": (tb if tb is not None else (traceback.format_exc() if e is not None else None)),
            }

    standardized_src = None

    try:
        with tempfile.TemporaryDirectory(prefix="mp_tmp_") as tmpd:
            tmp = Path(tmpd)
            trace_json_path = tmp / "trace.json"
            standardized_path = tmp / "standardized.py"

            if flag_parametricCurve:
                rec["stages"]["mode"] = "parametricCurve"

                orig_code = script_path.read_text(encoding="utf-8")
                code0 = del_lambda(orig_code) if "lambda" in orig_code else orig_code

                points = [25, 50, 100, 200]

                any_success = False
                last_standardized_saved = None

                for num_points in points:
                    attempt = {
                        "num_points": num_points,
                        "stages": {
                            "replace": False,
                            "record": False,
                            "standardize": False,
                            "exec_original": False,
                            "exec_standardized": False,
                            "check_shape": False,
                            "save_standardized": False,
                        },
                        "success": False,
                        "error": None,
                        "standardized_path": None,
                    }

                    def set_attempt_error(stage: str, e: Exception | None = None, msg: str | None = None):
                        if attempt["error"] is None:
                            attempt["error"] = {
                                "stage": stage,
                                "message": (msg if msg is not None else (str(e) if e is not None else "")),
                                "traceback": (traceback.format_exc() if e is not None else None),
                            }

                    standardized_src = None

                    try:
                        user_code = replace_parametric_curve(code0, num_points)
                        attempt["stages"]["replace"] = True
                    except Exception as e:
                        set_attempt_error("replace", e)
                        rec["attempts"].append(attempt)
                        continue

                    try:
                        LOG.clear()
                        patch()
                        ns_record = {}
                        exec(user_code, ns_record, ns_record)
                        attempt["stages"]["record"] = True
                    except Exception as e:
                        set_attempt_error("record", e)
                        rec["attempts"].append(attempt)
                        try:
                            unpatch()
                        except Exception:
                            pass
                        continue
                    finally:
                        try:
                            unpatch()
                        except Exception:
                            pass

                    try:
                        save_json(str(trace_json_path))
                        code = standardize_code(LOG, final_var="result")
                        save_code(str(standardized_path), code)
                        attempt["stages"]["standardize"] = True
                    except Exception as e:
                        set_attempt_error("standardize", e)
                        rec["attempts"].append(attempt)
                        continue

                    try:
                        ns_a = {}
                        exec(user_code, ns_a, ns_a)
                        a = ns_a.get("result")
                        attempt["stages"]["exec_original"] = True
                    except Exception as e:
                        set_attempt_error("exec_original", e)
                        rec["attempts"].append(attempt)
                        continue

                    try:
                        standardized_src = standardized_path.read_text(encoding="utf-8")
                        ns_b = {"np": np}
                        exec(standardized_src, ns_b, ns_b)
                        b = ns_b.get("result")
                        attempt["stages"]["exec_standardized"] = True
                    except Exception as e:
                        set_attempt_error("exec_standardized", e)
                        rec["attempts"].append(attempt)
                        continue

                    try:
                        c1, c2, c3 = check_shape_validity(a.val().wrapped)
                        ok_shape = bool(c1 and c2 and c3)
                        attempt["stages"]["check_shape"] = ok_shape
                    except Exception as e:
                        set_attempt_error("check_shape", e)
                        rec["attempts"].append(attempt)
                        continue

                    attempt["success"] = bool(attempt["stages"]["check_shape"])

                    if attempt["success"] and standardized_src is not None:
                        try:
                            standardized_dir.mkdir(parents=True, exist_ok=True)
                            dest_path = standardized_dir / f"{script_path.stem}_standardized_{num_points}.py"
                            save_code(str(dest_path), standardized_src)
                            attempt["stages"]["save_standardized"] = True
                            attempt["standardized_path"] = str(dest_path)
                            any_success = True
                            last_standardized_saved = str(dest_path)
                        except Exception as e:
                            set_attempt_error("save_standardized", e)
                            attempt["success"] = False

                    rec["attempts"].append(attempt)

                rec["success"] = any_success
                rec["stages"]["best_standardized_path"] = last_standardized_saved

                if not any_success and rec["error"] is None:
                    rec["error"] = {
                        "stage": "parametricCurve_all_attempts_failed",
                        "message": "All attempts are unsuccessful",
                        "traceback": None,
                    }

            else:
                import trimesh
                import gc
                from scipy.spatial import cKDTree

                rec["stages"]["mode"] = "standard"

                try:
                    user_code = script_path.read_text(encoding="utf-8")
                    rec["stages"]["read"] = True
                except Exception as e:
                    rec["stages"]["read"] = False
                    set_error("read", e)
                    return

                try:
                    LOG.clear()
                    patch()
                    ns_record = {}
                    exec(user_code, ns_record, ns_record)
                    rec["stages"]["record"] = True
                except Exception as e:
                    rec["stages"]["record"] = False
                    set_error("record", e)
                    return
                finally:
                    try:
                        unpatch()
                    except Exception:
                        pass

                try:
                    save_json(str(trace_json_path))
                    code = standardize_code(LOG, final_var="result")
                    save_code(str(standardized_path), code)
                    rec["stages"]["standardize"] = True
                except Exception as e:
                    rec["stages"]["standardize"] = False
                    set_error("standardize", e)
                    return

                try:
                    ns_a = {}
                    exec(user_code, ns_a, ns_a)
                    a = ns_a.get("result")
                    rec["stages"]["exec_original"] = True
                except Exception as e:
                    rec["stages"]["exec_original"] = False
                    set_error("exec_original", e)
                    return

                try:
                    standardized_src = standardized_path.read_text(encoding="utf-8")
                    ns_b = {"np": np}
                    exec(standardized_src, ns_b, ns_b)
                    b = ns_b.get("result")
                    rec["stages"]["exec_standardized"] = True
                except Exception as e:
                    rec["stages"]["exec_standardized"] = False
                    set_error("exec_standardized", e)
                    return

                try:
                    vertices_orig, triangles_orig = tessellate_solid(a.val().wrapped)
                    mesh_orig = trimesh.Trimesh(vertices=vertices_orig, faces=triangles_orig)

                    vertices_standardized, triangles_standardized = tessellate_solid(b.val().wrapped)
                    mesh_standardized = trimesh.Trimesh(vertices=vertices_standardized, faces=triangles_standardized)

                    extent_orig = mesh_orig.extents.max()
                    extent_standardized = mesh_standardized.extents.max()

                    mesh_standardized.apply_transform(trimesh.transformations.scale_matrix(1 / extent_standardized))
                    mesh_standardized.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

                    mesh_orig.apply_transform(trimesh.transformations.scale_matrix(1 / extent_orig))
                    mesh_orig.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

                    n_points = 20_000
                    gt_points, _ = trimesh.sample.sample_surface(mesh_orig, n_points)
                    pred_points, _ = trimesh.sample.sample_surface(mesh_standardized, n_points)

                    gt_points = gt_points.astype(np.float32, copy=False)
                    pred_points = pred_points.astype(np.float32, copy=False)

                    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
                    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)

                    gt_distance = gt_distance.astype(np.float32, copy=False)
                    pred_distance = pred_distance.astype(np.float32, copy=False)

                    cd = np.mean(gt_distance * gt_distance) + np.mean(pred_distance * pred_distance)

                    del mesh_orig, mesh_standardized, vertices_orig, triangles_orig, vertices_standardized, triangles_standardized
                    del gt_points, pred_points, gt_distance, pred_distance
                    gc.collect()

                    is_different = bool(cd > 0.15)
                    rec["stages"]["cd"] = float(cd)
                    rec["stages"]["compare"] = True
                    rec["is_different"] = is_different
                except Exception as e:
                    rec["stages"]["compare"] = False
                    set_error("compare", e)
                    return

                rec["success"] = (rec["is_different"] is False)

                if rec["success"] and standardized_src is not None:
                    try:
                        standardized_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = standardized_dir / f"{script_path.stem}_standardized.py"
                        save_code(str(dest_path), standardized_src)
                        rec["stages"]["save_standardized"] = True
                        rec["stages"]["standardized_path"] = str(dest_path)
                    except Exception as e:
                        rec["stages"]["save_standardized"] = False
                        set_error("save_standardized", e)
                        rec["success"] = False

    except Exception as e:
        set_error("process_detail", e)

    finally:
        if logs_path:
            try:
                log_line(str(logs_path), detail_name, json.dumps(rec, ensure_ascii=False), script=script_id)
            except Exception:
                pass


def run_standardizing(
    root_dir: Path,
    logs_path: Path,
    standardized_root: Path,
    n_workers: int = 16,
    task_timeout: int = 60,
    flat: bool = False,
):
    tasks = []

    root_dir = str(root_dir)
    standardized_root = str(standardized_root)
    logs_path = str(logs_path)

    if flat:
        already = set()
        if os.path.isdir(standardized_root):
            for f in os.scandir(standardized_root):
                if f.is_file() and f.name.endswith(".py"):
                    already.add(f.name)

        for file_entry in os.scandir(root_dir):
            if not file_entry.is_file():
                continue
            if not file_entry.name.endswith(".py"):
                continue

            stem = file_entry.name[:-3]
            standardized_name = f"{stem}_standardized.py"
            if standardized_name in already:
                continue

            content = Path(file_entry.path).read_text(encoding="utf-8")
            flag_parametric_curve = ("parametricCurve" in content)

            tasks.append({
                "detail": Path(file_entry.name).stem,
                "script_path": file_entry.path,
                "standardized_dir": standardized_root,
                "logs_path": logs_path,
                "flag_parametricCurve": flag_parametric_curve,
            })

    else:
        for detail_entry in os.scandir(root_dir):
            if not detail_entry.is_dir():
                continue

            detail_name = detail_entry.name
            detail_path = detail_entry.path

            scripts = []
            for f in os.scandir(detail_path):
                if f.is_file() and f.name.endswith(".py"):
                    scripts.append((f.name, f.path))

            if not scripts:
                continue

            standardized_dir = os.path.join(standardized_root, detail_name)

            existing_standardized = set()
            if os.path.isdir(standardized_dir):
                for ef in os.scandir(standardized_dir):
                    if ef.is_file() and ef.name.endswith("_standardized.py"):
                        existing_standardized.add(ef.name[:-len("_standardized.py")])

            if len(existing_standardized) >= 10:
                log_line(logs_path, detail_name, "skip: already complete (>=10 standardized)")
                continue

            pending = [(n, p) for (n, p) in scripts if (n[:-3] not in existing_standardized)]
            if not pending:
                log_line(logs_path, detail_name, "skip: nothing pending")
                continue

            for name, path in pending:
                content = Path(path).read_text(encoding="utf-8")
                flag_parametric_curve = ("parametricCurve" in content)

                tasks.append({
                    "detail": detail_name,
                    "script_path": path,
                    "standardized_dir": standardized_dir,
                    "logs_path": logs_path,
                    "flag_parametricCurve": flag_parametric_curve,
                })

    pool = ProcessPool(
        task_func=process_detail,
        task_args=tasks,
        n_processes=int(n_workers),
        timeout=int(task_timeout),
    )
    unprocessed_args = pool.run()

    log_line(
        logs_path,
        "SYSTEM",
        f"standardization pipeline done (tasks_total={len(tasks)}, unprocessed={len(unprocessed_args)})",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=Path, required=True)
    p.add_argument("--logs-path", type=Path, required=True)
    p.add_argument("--standardized-root", type=Path, required=True)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--task-timeout", type=int, default=60)
    p.add_argument("--flat", action="store_true")
    a = p.parse_args()

    a.standardized_root.mkdir(parents=True, exist_ok=True)
    a.logs_path.parent.mkdir(parents=True, exist_ok=True)

    run_standardizing(
        root_dir=a.root_dir,
        logs_path=a.logs_path,
        standardized_root=a.standardized_root,
        n_workers=a.n_workers,
        task_timeout=a.task_timeout,
        flat=a.flat,
    )
