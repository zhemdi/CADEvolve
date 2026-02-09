import ast
import re
from typing import Sequence, Tuple

import cadquery as cq
import numpy as np

# ---------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------
def cross(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def normalize_int(v):
    def n(c):
        if abs(c) < 0.5:
            return 0
        return 1 if c > 0 else -1
    return (n(v[0]), n(v[1]), n(v[2]))


class Rotator:
    """
    Rotates vectors by X/Y/Z rotations in {0,90,180,270}
    using explicit coordinate permutations.
    """

    BASES = {
        'XY': ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        'XZ': ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        'YZ': ((0, 1, 0), (0, 0, 1), (1, 0, 0)),

        'YX': ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        'ZX': ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        'ZY': ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    }
    

    def __init__(self, X_rot=0, Y_rot=0, Z_rot=0):
        self.X_rot = X_rot
        self.Y_rot = Y_rot
        self.Z_rot = Z_rot
        self.orientation_sign = 1

        self.face_selector_axes = {
        '>Y': ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        '+Y': ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        '<Y': (( 1, 0, 0), (0, 0, 1), (0,-1, 0)),
        'Y':  (( 1, 0, 0), (0, 0, 1), (0,-1, 0)),

        '>X': (( 0, 1, 0), (0, 0, 1), (1, 0, 0)),
        '+X': (( 0, 1, 0), (0, 0, 1), (1, 0, 0)),
        '<X': (( 0,-1, 0), (0, 0, 1), (-1, 0, 0)),
        'X':  (( 0,-1, 0), (0, 0, 1), (-1, 0, 0)),

        '>Z': (( 1, 0, 0), (0, 1, 0), (0, 0, 1)),
        '+Z': (( 1, 0, 0), (0, 1, 0), (0, 0, 1)),
        '<Z': (( 1, 0, 0), (0,-1, 0), (0, 0,-1)),
        'Z':  (( 1, 0, 0), (0,-1, 0), (0, 0,-1)),
        }


        self.face_axes: dict[str, Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]] = {}


    def rotate(self, x, y, z):
        # X
        if self.X_rot == 90:
            y, z = -z, y
        elif self.X_rot == 180:
            y, z = -y, -z
        elif self.X_rot == 270:
            y, z = z, -y

        # Y
        if self.Y_rot == 90:
            x, z = z, -x
        elif self.Y_rot == 180:
            x, z = -x, -z
        elif self.Y_rot == 270:
            x, z = -z, x

        # Z
        if self.Z_rot == 90:
            x, y = -y, x
        elif self.Z_rot == 180:
            x, y = -x, -y
        elif self.Z_rot == 270:
            x, y = y, -x

        return float(x), float(y), float(z)

    def plane_for(self, orientation: str, offset=(0.0, 0.0, 0.0)):

        # new plane
        self.orientation_sign = 1

        ori = orientation.upper()
        if ori not in self.BASES:
            ori = 'XY'

        x0, _, n0 = self.BASES[ori]

        # rotate basis
        x = self.rotate(*x0)
        n = self.rotate(*n0)

        # enforce right-handed basis
        y = cross(n, x)
        x = cross(y, n)

        x = normalize_int(x)
        n = normalize_int(n)

        if (self.X_rot > 90) ^ (self.Y_rot > 90) ^ (self.Z_rot > 90):
            self.orientation_sign *= -1

        # rotate offset (local → global)
        ox, oy, oz = offset
        origin = self.rotate(ox, oy, oz)

        return origin, x, n

    def rotate_selector_axis(self, selector: str) -> str | None:
        """
        Rotate axis selectors:
        |X |Y |Z
        +X -X +Y -Y +Z -Z
        >X <X >Y <Y >Z <Z

        Preserves selector semantics.
        """
        if len(selector) != 2:
            return None

        sign, axis = selector[0], selector[1]
        if axis not in 'XYZ':
            return None

        if sign not in '|+-<>':
            return None

        # base vector for axis
        vec = {
            'X': (1, 0, 0),
            'Y': (0, 1, 0),
            'Z': (0, 0, 1),
        }[axis]

        # physical sign for + / -
        if sign == '-' or sign == '<':
            vec = (-vec[0], -vec[1], -vec[2])

        # rotate vector
        rx, ry, rz = self.rotate(*vec)
        r = normalize_int((rx, ry, rz))

        axis_map = {
            (1, 0, 0): ('X', '+'),
            (-1, 0, 0): ('X', '-'),
            (0, 1, 0): ('Y', '+'),
            (0, -1, 0): ('Y', '-'),
            (0, 0, 1): ('Z', '+'),
            (0, 0, -1): ('Z', '-'),
        }

        if r not in axis_map:
            return None

        new_axis, phys_sign = axis_map[r]

        # rebuild selector
        if sign == '|':
            return '|' + new_axis

        if sign in '+-':
            return phys_sign + new_axis

        if sign in '><':
            if phys_sign == '-':
                sign = '<'
            else:
                sign = '>'
            # preserve semantic direction
            return sign + new_axis

        return None




# ---------------------------------------------------------------------
# AST Transformer
# ---------------------------------------------------------------------
class WorkplaneTransformer(ast.NodeTransformer):

    TARGET_FUNCS = {'Workplane', 'workplane', 'workplaneFromTagged'}

    SELECTOR_FUNCS = {'edges', 'faces', 'vertices', 'solids'}

    GLOBAL_COORD_FUNCS = {'rotate', 'translate','translateBy','translateTo',
                          'mirror', 'moveToGlobal', 'rotateAboutCenter',
                          'NearestToPointSelector'}


    def __init__(self, rotator: Rotator):
        super().__init__()
        self.rotator = rotator

    def _extract_numeric_value(self, node: ast.AST):
        """Extract numeric value from various AST node types"""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers: -5
            value = self._extract_numeric_value(node.operand)
            if value is not None:
                return -value
        
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            # Handle positive numbers: +5
            value = self._extract_numeric_value(node.operand)
            if value is not None:
                return value
        return None

    def _is_plane_call(self, node):
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == 'Plane'
        return False

    def _extract_vec3(self, node):
        # (x, y, z)
        if isinstance(node, (ast.Tuple, ast.List)):
            vals = []
            for e in node.elts:
                val = self._extract_numeric_value(e)
                if val is not None:
                    vals.append(val)
                else:
                    return None
            return tuple(vals[:3])

        # cq.Vector(x, y, z)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'Vector':
                vals = []
                for a in node.args:
                    val = self._extract_numeric_value(a)
                    if val is not None:
                        vals.append(val)
                    else:
                        return None
                return tuple(vals[:3])

        return None

    def _rotate_plane_call(self, plane_call):
        origin = (0.0, 0.0, 0.0)
        xdir = None
        normal = None

        for kw in plane_call.keywords:
            if kw.arg == 'origin':
                origin = self._extract_vec3(kw.value) or origin
            elif kw.arg == 'xDir':
                xdir = self._extract_vec3(kw.value)
            elif kw.arg == 'normal':
                normal = self._extract_vec3(kw.value)

        if not xdir or not normal:
            return plane_call  # unsafe to touch

        # rotate everything
        ox, oy, oz = self.rotator.rotate(*origin)
        x = self.rotator.rotate(*xdir)
        n = self.rotator.rotate(*normal)

        return ast.Call(
            func=plane_call.func,
            args=[],
            keywords=[
                ast.keyword('origin', _ast_tuple((ox, oy, oz))),
                ast.keyword('xDir', _ast_tuple(x)), # normalize_int(x)
                ast.keyword('normal', _ast_tuple(n)), # normalize_int(n)
            ]
        )
    
    def _handle_global_coord_func(self, node: ast.Call, func: str) -> ast.Call:
        """
        Rotate arguments of functions that operate in GLOBAL coordinates.
        """
        # rotate(p1, p2, angle)
        if func == 'rotate' and len(node.args) >= 2:
            p1 = self._extract_vec3(node.args[0])
            p2 = self._extract_vec3(node.args[1])
            if p1 and p2:
                node.args[0] = _ast_tuple(self.rotator.rotate(*p1))
                node.args[1] = _ast_tuple(self.rotator.rotate(*p2))
            return node

        # translate(vec)
        if func in ('translate', 'translateBy', 'translateTo', 'NearestToPointSelector'):
            # positional
            if node.args:
                v = self._extract_vec3(node.args[0])
                if v:
                    node.args[0] = _ast_tuple(self.rotator.rotate(*v))

            # keyword forms
            for kw in node.keywords:
                if kw.arg in ('vector', 'offset', 'origin', 'v'):
                    vec = self._extract_vec3(kw.value)
                    if vec:
                        kw.value = _ast_tuple(self.rotator.rotate(*vec))

            return node

        if func == 'mirror' and node.args:
            arg = node.args[0]

            # mirror(cq.Plane(...))
            if self._is_plane_call(arg):
                node.args[0] = self._rotate_plane_call(arg)
                return node

            # mirror(Vector | point)
            vec = self._extract_vec3(arg)
            if vec:
                node.args[0] = _ast_tuple(self.rotator.rotate(*vec))
                return node

            # keyword forms
            for kw in node.keywords:
                if kw.arg in ('plane', 'vector', 'point', 'origin'):
                    if self._is_plane_call(kw.value):
                        kw.value = self._rotate_plane_call(kw.value)
                    else:
                        vec = self._extract_vec3(kw.value)
                        if vec:
                            kw.value = _ast_tuple(self.rotator.rotate(*vec))
            return node

        if func == 'moveToGlobal':
            if len(node.args) == 1:
                vec = self._extract_vec3(node.args[0])
                if vec:
                    node.args[0] = _ast_tuple(self.rotator.rotate(*vec))
            elif len(node.args) >= 3:
                vec = self._extract_vec3(ast.Tuple(node.args[:3], ast.Load()))
                if vec:
                    rx, ry, rz = self.rotator.rotate(*vec)
                    node.args[0] = ast.Constant(rx)
                    node.args[1] = ast.Constant(ry)
                    node.args[2] = ast.Constant(rz)
            return node
        
        if func == 'rotateAboutCenter' and node.args:
            axis = self._extract_vec3(node.args[0])
            if axis:
                node.args[0] = _ast_tuple(self.rotator.rotate(*axis))
            return node

        return node

    def _get_kw(self, node, name):
        for kw in node.keywords:
            if kw.arg == name:
                return kw
        return None

    def plane_to_named_workplane(self, xDir, normal):
        """
        Return 'XY', 'XZ', 'YZ' if axes exactly match CadQuery base planes.
        Otherwise return None.
        """
        x = normalize_int(xDir)
        n = normalize_int(normal)

        if x == (1, 0, 0) and n == (0, 0, 1):
            return 'XY'
        
        if x == (0, 1, 0) and n == (0, 0, -1):
            return 'YX'

        if x == (1, 0, 0) and n == (0, -1, 0):
            return 'XZ'

        if x == (0, 0, 1) and n == (0, 1, 0):
            return 'ZX'

        if x == (0, 1, 0) and n == (1, 0, 0):
            return 'YZ'

        if x == (0, 0, 1) and n == (-1, 0, 0):
            return 'ZY'

        return None

    def visit_Call(self, node: ast.Call):
        if self._is_plane_call(node):
            return self._rotate_plane_call(node)
        
        self.generic_visit(node)
        func = self._func_name(node.func)

        # global functions
        if func in self.GLOBAL_COORD_FUNCS:
            node = self._handle_global_coord_func(node, func)
        
        # обработка revolve отдельно (особенности OCC)
        if func == 'revolve' and self.rotator.orientation_sign < 0:

            kw_start = self._get_kw(node, 'axisStart')
            kw_end   = self._get_kw(node, 'axisEnd')

            if kw_start and kw_end:
                # ось уже есть → просто меняем местами
                kw_start.value, kw_end.value = kw_end.value, kw_start.value

            elif len(node.args) >= 3 :
                # если ось есть как аргумент
                node.args[1], node.args[2] = node.args[2], node.args[1]

            else:
                node.keywords += [
                ast.keyword(arg='axisStart', value=_ast_tuple((0, 0))),
                ast.keyword(arg='axisEnd',   value=_ast_tuple((0, 1))),
                ]

        # selectors
        if func in self.SELECTOR_FUNCS and node.args:
            sel = node.args[0]
            if isinstance(sel, ast.Constant) and isinstance(sel.value, str):
                new_sel = self.rotator.rotate_selector_axis(sel.value)
                if new_sel:
                    node.args[0] = ast.Constant(value=new_sel)


        if func not in self.TARGET_FUNCS:
            return node

        plane_name = None
        offset = (0.0, 0.0, 0.0)
        new_keywords = []

        # positional "XY"
        if node.args and isinstance(node.args[0], ast.Constant):
            if isinstance(node.args[0].value, str):
                plane_name = node.args[0].value.upper()

        # keywords
        for kw in node.keywords:
            if kw.arg in ('origin', 'offset'):
                if isinstance(kw.value, (ast.Tuple, ast.List)):
                    vals = []
                    for e in kw.value.elts:
                        val = self._extract_numeric_value(e)
                        if val is not None:
                            vals.append(val)
                    if len(vals) == 2:
                        vals.append(0.0)
                    if len(vals) >= 3:
                        offset = tuple(vals[:3])
            else:
                new_keywords.append(kw)

        if plane_name and plane_name in self.rotator.BASES:
            origin, xdir, normal = self.rotator.plane_for(plane_name, offset)

            wp_name = self.plane_to_named_workplane(xdir, normal)

            if wp_name:
                # workplane('XY'|'XZ'|'YZ', origin=...)
                return ast.copy_location(
                    ast.Call(
                        func=node.func,
                        args=[ast.Constant(value=wp_name)],
                        keywords=[
                            ast.keyword(arg='origin', value=_ast_tuple(origin)),
                            *new_keywords
                        ]
                    ),
                    node
                )

            plane_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='cq', ctx=ast.Load()),
                    attr='Plane',
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[
                    ast.keyword(arg='origin', value=_ast_tuple(origin)),
                    ast.keyword(arg='xDir', value=_ast_tuple(xdir)),
                    ast.keyword(arg='normal', value=_ast_tuple(normal)),
                ]
            )

            return ast.copy_location(
                ast.Call(
                    func=node.func,
                    args=[plane_call],
                    keywords=new_keywords
                ),
                node
            )
        

        return node

    @staticmethod
    def _func_name(fn):
        if isinstance(fn, ast.Name):
            return fn.id
        if isinstance(fn, ast.Attribute):
            return fn.attr
        return None


def _ast_tuple(seq: Sequence[float]) -> ast.Tuple:
    return ast.Tuple(
        elts=[ast.Constant(value=v) for v in seq],
        ctx=ast.Load()
    )


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
def transform_source(source: str, rotator: Rotator) -> str:
    tree = ast.parse(source)
    tree = WorkplaneTransformer(rotator).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)