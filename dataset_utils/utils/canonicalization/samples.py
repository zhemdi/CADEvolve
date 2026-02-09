import json
import random
import numpy as np
import ast
import multiprocessing as mp
import signal
import argparse
from pathlib import Path
from inspect import signature
from multiprocessing import Process
from multiprocessing.pool import Pool
from multiprocessing import get_context
from tqdm import tqdm
from scipy.spatial import cKDTree
from utils.canonicalization.helpers import tessellate_solid

#========== Per-task timeout helpers ==========


class TaskTimeout(Exception):
    pass


def _task_timeout_handler(signum, frame):
    raise TaskTimeout()


# ========== Auxiliary functions for shape validation ==========


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


# ========== Shape validation ==========


def check_shape_validity(topo_shape):
    try:
        top_solids = extract_subsolids(topo_shape)
        if len(top_solids) == 0:
            c1 = False
        elif len(top_solids) == 1:
            c1 = True
        else:
            bbox = cq.Shape.cast(topo_shape).BoundingBox()
            tol = max(bbox.xlen, bbox.ylen, bbox.zlen) * 0.01  # 1% of largest dimension
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


# ========== Main function to create a script from code for a given parameter set using AST transformations ==========


def create_script(code: str, param_set: dict = None):
    try_block_counter = 0

    _REDUNDANT_CASTS = {"int", "float"}

    def _is_redundant_cast(stmt: ast.Assign) -> bool:
        if not (isinstance(stmt.value, ast.Call) and
                isinstance(stmt.targets[0], ast.Name)):
            return False

        target_id = stmt.targets[0].id
        call = stmt.value

        return (
            isinstance(call.func, ast.Name) and
            call.func.id in _REDUNDANT_CASTS and
            len(call.args) == 1 and
            isinstance(call.args[0], ast.Name) and
            call.args[0].id == target_id
        )
        
    def handle_try_except(try_node: ast.Try,
                        params: dict,
                        script: list,
                        eval_context: dict,
                        for_flag: bool = False) -> bool:
        nonlocal try_block_counter

        if for_flag:
            namespace = {**eval_context, **params}

            try_block = ast.Module(body=try_node.body, type_ignores=[])
            try_code = compile(ast.fix_missing_locations(try_block), filename="<ast>", mode="exec")

            try:
                exec(try_code, namespace, namespace)
            except Exception:
                if try_node.handlers:
                    handler_block = ast.Module(body=try_node.handlers[0].body, type_ignores=[])
                    handler_code = compile(ast.fix_missing_locations(handler_block), filename="<ast>", mode="exec")
                    exec(handler_code, namespace, namespace)

            update_context_from_namespace(namespace)
            script.append(ast.fix_missing_locations(try_node))
            return False

        var_name = f"try_result_{try_block_counter}"
        try_block_counter += 1
        try_node.body.append(
            ast.Assign([ast.Name(var_name, ast.Store())], ast.Constant("pass"))
        )

        if try_node.handlers:
            h = try_node.handlers[0]
            new_body = []
            for s in h.body:
                if isinstance(s, ast.Return):
                    if not (isinstance(s.value, ast.Name) and s.value.id == "result"):
                        new_body.append(
                            ast.Assign([ast.Name("result", ast.Store())], s.value)
                        )
                    h.body = [ast.Assign([ast.Name(var_name, ast.Store())],
                                         ast.Constant("error"))] + new_body
                    script.append(ast.fix_missing_locations(try_node))
                    return True
                new_body.append(s)
            h.body = [ast.Assign([ast.Name(var_name, ast.Store())],
                                 ast.Constant("error"))] + new_body

        script.append(ast.fix_missing_locations(try_node))
        return False
        
    def _body_has_only_raise(block: list) -> bool:
        """Check if a block contains only raise statements (and docstring literals)."""
        for st in block:
            if isinstance(st, ast.Raise):
                continue
            if (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str)):
                continue
            return False
        return True

    def handle_ann_assign(ann_assign_stmt, body):
        try:
            if ann_assign_stmt.value is not None:
                expr = compile(ast.Expression(ann_assign_stmt.value), "", "eval")
                namespace = {**eval_context, **params}
                value = eval(expr, namespace, namespace)
                var_name = ann_assign_stmt.target.id
                params[var_name] = value
                eval_context[var_name] = value
        except Exception:
            pass
        if not _is_redundant_cast(ann_assign_stmt):
            body.append(ast.fix_missing_locations(ann_assign_stmt))

    def handle_assign(assign_stmt, body):
        try:
            expr = compile(ast.Expression(assign_stmt.value), "", "eval")
            namespace = {**eval_context, **params}
            value = eval(expr, namespace, namespace)
            update_context_from_namespace(namespace)
            target = assign_stmt.targets[0]
            if isinstance(target, ast.Name):
                params[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (list, tuple)):
                for var_node, val in zip(target.elts, value):
                    if isinstance(var_node, ast.Name):
                        params[var_node.id] = val
        except Exception:
            pass
        if not _is_redundant_cast(assign_stmt):
            body.append(ast.fix_missing_locations(assign_stmt))

    ALLOWED = (
    int, float, str, bool, list, tuple, dict,
    cq.Workplane, cq.Shape, cq.Vector, cq.Location, cq.Plane,
    )

    def update_context_from_namespace(ns):
        for k, v in ns.items():
            if not k.startswith("__") and isinstance(v, ALLOWED):
                params[k] = v
        eval_context.update(ns)

    def evaluate_expr_statement(stmt, container):
        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            return
        try:
            namespace = {**eval_context, **params}
            code_obj = compile(ast.Expression(stmt.value), "", "eval")
            eval(code_obj, namespace, namespace)
            update_context_from_namespace(namespace)
        except Exception:
            pass
        container.append(ast.fix_missing_locations(stmt))

    def handle_function_def(node, container):
        for i, subnode in enumerate(node.body):
            if isinstance(subnode, ast.Nonlocal):
                node.body[i] = ast.Global(names=subnode.names)
        try:
            mod = ast.Module(body=[node], type_ignores=[])
            code_obj = compile(ast.fix_missing_locations(mod), "", "exec")
            exec(code_obj, eval_context, params)
        except Exception:
            pass
        container.append(ast.fix_missing_locations(node))

    def process_if_in_for_node(node: ast.If, params: dict, new_body: list, loop_vars: set, assigned_vars: dict):

        condition_code = ast.unparse(node.test)
        used_vars = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        namespace = {**eval_context, **params}
        processed_body: list = []
        def __is_dependent(var, visited=None):
            if visited is None:
                visited = set()
            if var in visited:
                return False
            visited.add(var)
            if var in loop_vars:
                return True
            if var in assigned_vars:
                return True
            parents = assigned_vars.get(var, set())
            return any(__is_dependent(p, visited) for p in parents)

        def __block_is_only_nonconstructive_raise(block):
            return all(isinstance(stmt, ast.Raise) for stmt in block)


        def __process_assign(stmt, body_list):
            try:
                expr = compile(ast.Expression(stmt.value), "", "eval")
                value = eval(expr, namespace, namespace)
                if isinstance(stmt.targets[0], ast.Name):
                    params[stmt.targets[0].id] = value
                elif isinstance(stmt.targets[0], (ast.Tuple, ast.List)) and isinstance(value, (list, tuple)):
                    for var_node, val in zip(stmt.targets[0].elts, value):
                        if isinstance(var_node, ast.Name):
                            params[var_node.id] = val
                used = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
                if isinstance(stmt.targets[0], ast.Name):
                    assigned_vars.setdefault(stmt.targets[0].id, set()).update(used)
                elif isinstance(stmt.targets[0], (ast.Tuple, ast.List)):
                    for var_node in stmt.targets[0].elts:
                        if isinstance(var_node, ast.Name):
                            assigned_vars.setdefault(var_node.id, set()).update(used)
                update_context_from_namespace(namespace)
            except Exception:
                pass
            if not _is_redundant_cast(stmt):
                body_list.append(ast.fix_missing_locations(stmt))

        if any(__is_dependent(var) for var in used_vars):
            if __block_is_only_nonconstructive_raise(node.body):
                return

            try:
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr):
                        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                            continue
                        if isinstance(stmt.value, ast.Call):
                            func = stmt.value.func
                            if isinstance(func, ast.Attribute) and func.attr == "append":
                                if stmt.value.args:
                                    used_in_append = {n.id for n in ast.walk(stmt.value.args[0]) if isinstance(n, ast.Name)}
                                    if any(v in loop_vars for v in used_in_append):
                                        processed_body.append(ast.fix_missing_locations(stmt))
                                        continue
                        try:
                            code_obj = compile(ast.Expression(stmt.value), "", "eval")
                            eval(code_obj, namespace, namespace)
                            update_context_from_namespace(namespace)
                        except Exception:
                            pass
                        processed_body.append(ast.fix_missing_locations(stmt))
                    elif isinstance(stmt, ast.Assign):
                        __process_assign(stmt, processed_body)
                    elif isinstance(stmt, ast.If):
                        temp_body = []
                        process_if_in_for_node(stmt, params, temp_body, loop_vars, assigned_vars)
                        if temp_body:
                            processed_body.extend(temp_body)
                        else:
                            processed_body.append(ast.fix_missing_locations(stmt))
                    elif isinstance(stmt, ast.Return):
                        if isinstance(stmt.value, ast.Name) and stmt.value.id == "result":
                            continue
                        assign_node = ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=stmt.value)
                        processed_body.append(ast.fix_missing_locations(assign_node))
                        return
                    elif isinstance(stmt, ast.Raise):
                        continue
                    elif isinstance(stmt, ast.For):
                        process_for_node(stmt, params, processed_body, eval_context, loop_vars)
                    elif isinstance(stmt, ast.FunctionDef):
                        handle_function_def(stmt, processed_body)
                    elif isinstance(stmt, ast.ClassDef):
                        processed_body.append(ast.fix_missing_locations(stmt))
                    elif isinstance(stmt, ast.AnnAssign):
                        handle_ann_assign(stmt, processed_body)
                    elif isinstance(stmt, ast.Try):
                        handle_try_except(stmt, params, processed_body, eval_context, for_flag=True)
                    else:
                        processed_body.append(ast.fix_missing_locations(stmt))
                new_if = ast.If(test=node.test, body=processed_body, orelse=node.orelse or [])
                new_body.append(ast.fix_missing_locations(new_if))
            except Exception:
                new_if = ast.If(test=node.test, body=processed_body, orelse=node.orelse or [])
                new_body.append(ast.fix_missing_locations(new_if))
            return

        try:
            if eval(condition_code, namespace, namespace):
                block = node.body
            else:
                if _body_has_only_raise(node.body):
                    raise ValueError(f"[BAD SAMPLING]\n Condition: {condition_code}\n Parameters: {params_from_func}\n")
                block = node.orelse or []
            for stmt in block:
                if isinstance(stmt, ast.Expr):
                    evaluate_expr_statement(stmt, new_body)
                elif isinstance(stmt, ast.Assign):
                    __process_assign(stmt, new_body)
                elif isinstance(stmt, ast.If):
                    process_if_in_for_node(stmt, params, new_body, loop_vars, assigned_vars)
                elif isinstance(stmt, ast.Return):
                    if isinstance(stmt.value, ast.Name) and stmt.value.id == "result":
                        continue
                    assign_node = ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=stmt.value)
                    new_body.append(ast.fix_missing_locations(assign_node))
                    return
                elif isinstance(stmt, ast.Raise):
                    continue
                elif isinstance(stmt, ast.For):
                    process_for_node(stmt, params, new_body, eval_context)
                elif isinstance(stmt, ast.AnnAssign):
                    handle_ann_assign(stmt, new_body)
                elif isinstance(stmt, ast.FunctionDef):
                    handle_function_def(stmt, new_body)
                elif isinstance(stmt, ast.ClassDef):
                    new_body.append(ast.fix_missing_locations(stmt))
                elif isinstance(stmt, ast.Try):
                    handle_try_except(stmt, params, new_body, eval_context, for_flag=True)
                else:
                    new_body.append(ast.fix_missing_locations(stmt))
            return
        except Exception:
            new_if = ast.If(test=node.test, body=processed_body or node.body, orelse=node.orelse or [])
            new_body.append(ast.fix_missing_locations(new_if))

    def process_if_node(node, params, script) -> bool:
        condition_code = ast.unparse(node.test)
        namespace = {**eval_context, **params}
        try:
            if eval(condition_code, namespace, namespace):
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        if isinstance(stmt.value, ast.Name) and stmt.value.id == "result":
                            continue
                        script.append(ast.fix_missing_locations(ast.Assign(
                            targets=[ast.Name(id="result", ctx=ast.Store())],
                            value=stmt.value
                        )))
                        return True
                    elif isinstance(stmt, ast.If):
                        if process_if_node(stmt, params, script):
                            return True
                    elif isinstance(stmt, ast.Raise):
                        continue
                    elif isinstance(stmt, ast.Assign):
                        handle_assign(stmt, script)
                    elif isinstance(stmt, ast.For):
                        process_for_node(stmt, params, script, eval_context)
                    elif isinstance(stmt, ast.FunctionDef):
                        handle_function_def(stmt, script)
                    elif isinstance(stmt, ast.ClassDef):
                        script.append(ast.fix_missing_locations(stmt))
                    elif isinstance(stmt, ast.AnnAssign):
                        handle_ann_assign(stmt, script)
                    elif isinstance(stmt, ast.Expr):
                        evaluate_expr_statement(stmt, script)
                    elif isinstance(stmt, ast.Try):
                        handle_try_except(stmt, params, script, eval_context)
                    else:
                        script.append(ast.fix_missing_locations(stmt))
            else:
                if _body_has_only_raise(node.body):
                    raise ValueError(f"[BAD SAMPLING]\n Condition: {condition_code}\n Parameters: {params_from_func}\n")
                if node.orelse:
                    for stmt in node.orelse:
                        if isinstance(stmt, ast.If):
                            if process_if_node(stmt, params, script):
                                return True
                        elif isinstance(stmt, ast.Assign):
                            handle_assign(stmt, script)
                        elif isinstance(stmt, ast.For):
                            process_for_node(stmt, params, script, eval_context)
                        elif isinstance(stmt, ast.Return):
                            if isinstance(stmt.value, ast.Name) and stmt.value.id == "result":
                                continue
                            script.append(ast.fix_missing_locations(ast.Assign(
                                targets=[ast.Name(id="result", ctx=ast.Store())],
                                value=stmt.value
                            )))
                            return True
                        elif isinstance(stmt, ast.Raise):
                            continue
                        elif isinstance(stmt, ast.FunctionDef):
                            handle_function_def(stmt, script)
                        elif isinstance(stmt, ast.ClassDef):
                            script.append(ast.fix_missing_locations(stmt))
                        elif isinstance(stmt, ast.AnnAssign):
                            handle_ann_assign(stmt, script)
                        elif isinstance(stmt, ast.Expr):
                            evaluate_expr_statement(stmt, script)
                        elif isinstance(stmt, ast.Try):
                            handle_try_except(stmt, params, script, eval_context)
                        else:
                            script.append(ast.fix_missing_locations(stmt))
        except Exception:
            pass
        return False

    def process_while_node(item: ast.While, params: dict, script: list, eval_context: dict):
        new_body = []
        loop_vars = set()
        namespace = {**eval_context, **params}
        for node in ast.walk(item.test):
            if isinstance(node, ast.Name):
                loop_vars.add(node.id)
        assigned_vars = {}
        for stmt in item.body:
            if isinstance(stmt, ast.Expr):
                evaluate_expr_statement(stmt, new_body)
            elif isinstance(stmt, ast.Assign):
                handle_assign(stmt, new_body)
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    deps = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
                    assigned_vars[target.id] = deps
                elif isinstance(target, (ast.Tuple, ast.List)):
                    deps = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
                    for var_node in target.elts:
                        if isinstance(var_node, ast.Name):
                            assigned_vars.setdefault(var_node.id, set()).update(deps)
            elif isinstance(stmt, ast.If):
                process_if_in_for_node(stmt, params, new_body, loop_vars, assigned_vars)
            elif isinstance(stmt, ast.For):
                process_for_node(stmt, params, new_body, eval_context)
            elif isinstance(stmt, ast.While):
                process_while_node(stmt, params, new_body, eval_context)
            elif isinstance(stmt, ast.ClassDef):
                new_body.append(ast.fix_missing_locations(stmt))
            elif isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Name) and stmt.value.id == "result":
                    continue
                new_body.append(ast.fix_missing_locations(ast.Assign(
                    targets=[ast.Name(id="result", ctx=ast.Store())],
                    value=stmt.value
                )))
            elif isinstance(stmt, ast.AnnAssign):
                handle_ann_assign(stmt, new_body)
            elif isinstance(stmt, ast.FunctionDef):
                handle_function_def(stmt, new_body)
            elif isinstance(stmt, ast.Try):
                handle_try_except(stmt, params, new_body, eval_context, for_flag=True)
            else:
                new_body.append(ast.fix_missing_locations(stmt))

        if new_body:
            new_while = ast.fix_missing_locations(ast.While(test=item.test, body=new_body, orelse=item.orelse or []))
            script.append(new_while)

        try:
            exec_node = ast.Module(body=[new_while], type_ignores=[])
            code_obj = compile(ast.fix_missing_locations(exec_node), filename="<ast>", mode="exec")
            exec(code_obj, namespace, namespace)
            update_context_from_namespace(namespace)
        except Exception:
            pass

    def process_for_node(item: ast.For, params: dict, script: list, eval_context: dict, loop_vars=None):
        if loop_vars is None:
            loop_vars = set()
        current_loop_vars = set()
        if isinstance(item.target, ast.Name):
            current_loop_vars.add(item.target.id)
        elif isinstance(item.target, (ast.Tuple, ast.List)):
            for elt in item.target.elts:
                if isinstance(elt, ast.Name):
                    current_loop_vars.add(elt.id)
        combined_loop_vars = loop_vars | current_loop_vars
        namespace = {**eval_context, **params}
        try:
            iter_value = eval(compile(ast.Expression(item.iter), "", "eval"), namespace, namespace)
            iterator = iter(iter_value) if iter_value is not None else []
            first_element = next(iterator, None)
            if first_element is not None:
                if isinstance(item.target, ast.Tuple) and isinstance(first_element, (list, tuple)):
                    for elt, val in zip(item.target.elts, first_element):
                        if isinstance(elt, ast.Name) and elt.id not in params:
                            params[elt.id] = val
                elif isinstance(item.target, ast.Name):
                    if item.target.id not in params:
                        params[item.target.id] = first_element
        except Exception:
            pass

        new_body = []
        assigned_vars = {}
        for stmt in item.body:
            if isinstance(stmt, ast.Assign):
                handle_assign(stmt, new_body)
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    deps = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
                    assigned_vars[target.id] = deps
                elif isinstance(target, (ast.Tuple, ast.List)):
                    deps = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
                    for var_node in target.elts:
                        if isinstance(var_node, ast.Name):
                            assigned_vars.setdefault(var_node.id, set()).update(deps)
            elif isinstance(stmt, ast.Expr):
                evaluate_expr_statement(stmt, new_body)
            elif isinstance(stmt, ast.If):
                process_if_in_for_node(stmt, params, new_body, combined_loop_vars, assigned_vars)
            elif isinstance(stmt, ast.For):
                process_for_node(stmt, params, new_body, eval_context, loop_vars=combined_loop_vars)
            elif isinstance(stmt, ast.FunctionDef):
                handle_function_def(stmt, new_body)
            elif isinstance(stmt, ast.ClassDef):
                new_body.append(ast.fix_missing_locations(stmt))
            elif isinstance(stmt, ast.AnnAssign):
                handle_ann_assign(stmt, new_body)
            elif isinstance(stmt, ast.Try):
                handle_try_except(stmt, params, new_body, eval_context, for_flag=True)
            else:
                new_body.append(ast.fix_missing_locations(stmt))
        if not new_body:
            return
        new_loop = ast.fix_missing_locations(ast.For(target=item.target, iter=item.iter, body=new_body, orelse=item.orelse, type_comment=None))
        script.append(new_loop)
        try:
            exec_node = ast.Module(body=[new_loop], type_ignores=[])
            code_obj = compile(ast.fix_missing_locations(exec_node), filename="<ast>", mode="exec")
            exec(code_obj, namespace, namespace)
            update_context_from_namespace(namespace)
        except Exception:
            pass

    tree = ast.parse(code)
    script: list = []
    safe_builtins = {
        "abs": abs, "round": round, "max": max, "min": min, "pow": pow,
        "sum": sum, "len": len, "all": all, "any": any, "sorted": sorted,
        "range": range, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "list": list, "tuple": tuple, "int": int, "float": float, "str": str, "bool": bool,
        "isinstance": isinstance, "set": set, "random": random, "np": np,
        "Voronoi": Voronoi, "Vector": Vector, 
        "Plane": Plane, "Workplane": Workplane, "Shape": Shape,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "NameError": NameError, "IndexError": IndexError, "KeyError": KeyError,
        "AttributeError": AttributeError, "RuntimeError": RuntimeError, "ImportError": ImportError,
        "ZeroDivisionError": ZeroDivisionError, "StopIteration": StopIteration, 
        "AssertionError": AssertionError, "NotImplementedError": NotImplementedError, "OSError": OSError,
        "hasattr": hasattr, "reversed": reversed, "__import__": __import__
    }
    eval_context = {"__builtins__": safe_builtins, "math": math, "cq": cq}
    params: dict = {}

    func_def = None
    for item in tree.body:
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            script.append(ast.fix_missing_locations(item))
        if isinstance(item, ast.FunctionDef):
            func_def = item
            break
    if func_def is None:
        raise ValueError("No function definition found in the provided code.")

    base_param_values = {}
    defaults = list(func_def.args.defaults)
    total_args = func_def.args.args
    non_default_count = len(total_args) - len(defaults)
    for i, arg in enumerate(total_args):
        param_name = arg.arg
        if i < non_default_count:
            base_param_values[param_name] = None
        else:
            default_node = defaults[i - non_default_count]
            try:
                value = ast.literal_eval(default_node)
            except Exception:
                value = eval(compile(ast.Expression(default_node), "", "eval"), eval_context, {})
            base_param_values[param_name] = value

    params_from_func = base_param_values.copy()
    if param_set is not None:
        for arg in func_def.args.args:
            param_name = arg.arg
            val = param_set.get(param_name, base_param_values.get(param_name))
            params[param_name] = val
            val_node = ast.parse(repr(val), mode='eval').body
            script.append(ast.fix_missing_locations(ast.Assign(
                targets=[ast.Name(id=param_name, ctx=ast.Store())],
                value=val_node
            )))
    else:
        for arg, default in zip(func_def.args.args, func_def.args.defaults):
            params[arg.arg] = base_param_values[arg.arg]
            script.append(ast.fix_missing_locations(ast.Assign(
                targets=[ast.Name(id=arg.arg, ctx=ast.Store())],
                value=default
            )))
    imports = []


    for i, item in enumerate(func_def.body):
        node_type = type(item).__name__

        if isinstance(item, (ast.Import, ast.ImportFrom)):
            imports.append(ast.fix_missing_locations(item))
        elif isinstance(item, ast.If):
            stop = process_if_node(item, params, script)
            if stop:
                break
        elif isinstance(item, ast.FunctionDef):
            handle_function_def(item, script)
        elif isinstance(item, ast.Return):
            if isinstance(item.value, ast.Name) and item.value.id == "result":
                continue

            script.append(ast.fix_missing_locations(ast.Assign(
                targets=[ast.Name(id="result", ctx=ast.Store())],
                value=item.value
            )))
        elif isinstance(item, ast.Assign):
            handle_assign(item, script)
        elif isinstance(item, ast.For):
            process_for_node(item, params, script, eval_context)
        elif isinstance(item, ast.Expr):
            evaluate_expr_statement(item, script)
        elif isinstance(item, ast.ClassDef):
            try:
                mod = ast.Module(body=[item], type_ignores=[])
                code_obj = compile(ast.fix_missing_locations(mod), "<class_def>", "exec")
                exec(code_obj, eval_context, params)
            except Exception:
                pass
            script.append(ast.fix_missing_locations(item))
        elif isinstance(item, ast.While):
            process_while_node(item, params, script, eval_context)
        elif isinstance(item, ast.AnnAssign):
            handle_ann_assign(item, script)
        elif isinstance(item, ast.Try):
            should_stop = handle_try_except(item, params, script, eval_context)
            if should_stop:
                break
        else:
            script.append(ast.fix_missing_locations(item))
    module = ast.Module(body=imports + script, type_ignores=[])
    final_code = ast.unparse(module)
    return final_code


# ========== Post-processing for created script ==========


def ensure_required_imports(script_code: str) -> str:
    lines = script_code.splitlines()
    has_cq = any("import cadquery as cq" in line for line in lines)
    has_math_usage = "math." in script_code
    has_math_import = any("import math" in line for line in lines)

    import_lines = []

    if not has_cq:
        import_lines.append("import cadquery as cq")
    if has_math_usage and not has_math_import:
        import_lines.append("import math")

    if import_lines:
        script_code = "\n".join(import_lines) + "\n" + script_code

    return script_code


def remove_trailing_dot_zero(code: str) -> str:
    return re.sub(r'(?<![\w.])(\d+)\.0(?!\d)', r'\1', code)


class CollapseTry(ast.NodeTransformer):
    def __init__(self, ns):
        keys = [k for k in ns if k.startswith("try_result_")]
        self.results = {k: ns[k] for k in keys}
        self.queue   = iter(sorted(keys, key=lambda s: int(s.split("_")[-1])))
        
    def visit_FunctionDef(self, node):
        return node

    def visit_For(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("try_result_")):
            return None
        return node

    def visit_Try(self, node: ast.Try):
        node = self.generic_visit(node)

        try_name = next(self.queue, None)
        status   = self.results.get(try_name)

        if status == "pass":
            body = node.body

        elif status == "error" and node.handlers:
            handler_body = [
                st for st in node.handlers[0].body
                if not isinstance(st, (ast.Pass, ast.Raise)) and
                   not (isinstance(st, ast.Expr) and
                        isinstance(st.value, ast.Constant) and
                        isinstance(st.value.value, str))
            ]
            body = handler_body

        else:
            body = []

        return body or None


def collapse_script(source_code: str, ns: dict) -> str:
    tree   = ast.parse(source_code)
    tree   = CollapseTry(ns).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ========== Processing helpers ==========


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


# ========== Main helpers ==========


def load_code_db(json_path: str):
    with open(json_path, 'r') as f:
        return json.load(f)


def wp_is_empty(wp) -> bool:
    try:
        bbox = wp.val().BoundingBox()
        return bbox.xlen == 0 and bbox.ylen == 0 and bbox.zlen == 0
    except Exception:
        return True


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def compute_cd(pred_mesh, gt_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd


def meshes_differ(mesh_1, mesh_2, tol=0.15) -> bool:
    return compute_cd(mesh_1, mesh_2) < tol


def save_script_to_file(part_name: str, index: int, script_code: str, num_samples: int):
    part_folder = SAVE_DIR / part_name
    part_folder.mkdir(parents=True, exist_ok=True)

    width = max(2, len(str(num_samples)))
    script_path = part_folder / f"{index+1:0{width}d}.py"

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_code)


def append_txt_line(path: str, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


# ========== Covariance Matrix Adaptation - Evolution Strategy (CMA-ES) ==========


def classify(k, v):
    if v is None:
        return "none_int"
    if isinstance(v, (tuple, list)) and all(isinstance(x, (float, int)) for x in v):
        inner_types = [classify(f"{k}_{i}", x) for i, x in enumerate(v)]
        return f"tuple_{len(v)}:" + ",".join(inner_types)
    if any(sub in k.lower() for sub in ("angle", "theta", "phi", "deg", "degree")) and not any(sub in k.lower() for sub in ("frac", "fraction")):
        return "angle"
    if isinstance(v, str):
        return "str"
    if isinstance(v, int) or (isinstance(v, float) and v%1 == 0):
        return "int"
    return "pos" if v >= 0 else "free"


def enc(t, x):
    if t.startswith("tuple_"):
        _, type_str = t.split(":")
        subtypes = type_str.split(",")
        return [enc(st, xi) for st, xi in zip(subtypes, x)]
    if t == "none_int":
        return 0.0
    if t in ("pos", "int"):
        return math.log(x + 1e-15)
    if t == "angle":
        return (x % 360) / 360.0
    return x


def dec(t, z, precision=1):     
    if t.startswith("tuple_"):
        _, type_str = t.split(":")
        subtypes = type_str.split(",")
        return tuple(dec(st, zi, precision) for st, zi in zip(subtypes, z))
    if t == "pos":
        return round(math.exp(z) - 1e-15, precision)
    if t == "angle":
        return int(round((z % 1.0) * 360.0))
    if t == "int":
        return int(round(math.exp(z) - 1e-15))
    if t == "none_int":
        return None if z < 0.5 else max(1, int(round(z)))
    return round(z, precision) if isinstance(z, float) else z


def decode_z(z, types, keys, precisions):
    out = {}
    i = 0
    for k, t in zip(keys, types):
        p = precisions.get(k, 1)
        if t.startswith("tuple_"):
            _, type_str = t.split(":")
            n = int(t.split("_")[1].split(":")[0])
            subtypes = type_str.split(",")
            out[k] = tuple(dec(st, z[i+j], p) for j, st in enumerate(subtypes))
            i += n
        else:
            out[k] = dec(t, z[i], p)
            i += 1
    return out


def get_precision(v: float, max_prec: int = 16) -> int:
    if not isinstance(v, float):
        return 0

    for p in range(max_prec + 1):
        if round(v, p) == v:
            return p
    return max_prec   


def distance(a, b):
    return math.sqrt(sum((ai-bi)**2 for ai,bi in zip(a,b)))


def as_shape(obj):
    return obj.val() if hasattr(obj, 'val') else obj


def update_factor_scale(factor, key='ext'):
    if key == 'corner':
        factor-= 1
        return factor
    factor += 1
    return factor


def validate(shape, mesh=None, flag=True):
    c1, c2, c3 = check_shape_validity(shape.val().wrapped)
    if not (c1 and c2 and c3):
        return 1e3
    
    if flag:
        bb = as_shape(shape).BoundingBox()
        ext = max(bb.xmax-bb.xmin, bb.ymax-bb.ymin, bb.zmax-bb.zmin)
        penalty = 0
        penalty_term_1 = max(ext-200, 60-ext, 0)
        penalty += penalty_term_1

        penalty_term_2 = 0
        for v in [bb.xmin, bb.ymin, bb.zmin]: penalty_term_2 += max(0, -100-v)
        for v in [bb.xmax, bb.ymax, bb.zmax]: penalty_term_2 += max(0,  v-100)
        penalty += penalty_term_2
        return penalty
    
    mins, maxs = mesh.bounds
    xmin, ymin, zmin = map(float, mins)
    xmax, ymax, zmax = map(float, maxs)
    ext = max(xmax-xmin, ymax-ymin, zmax-zmin)
    penalty = 0
    penalty_term_1 = max(ext-200, 60-ext, 0)
    penalty += penalty_term_1

    penalty_term_2 = 0
    for v in [xmin, ymin, zmin]: penalty_term_2 += max(0, -100-v)
    for v in [xmax, ymax, zmax]: penalty_term_2 += max(0,  v-100)
    penalty += penalty_term_2

    return penalty


def rescale_shape(generator, defaults):
    factor_scale_corner = 100
    factor_scale_ext = 100
    orig_prec = {k: get_precision(v) for k, v in defaults.items()}
    flag = True
    penalty = float("inf")

    attempt = 0
    n_attempts = 10

    best_penalty = float("inf")
    best_result = None

    while attempt < n_attempts and penalty > 1e-6:
        attempt += 1
        shape = generator(**defaults)
        bb = as_shape(shape).BoundingBox()
        ext = max(bb.xmax-bb.xmin, bb.ymax-bb.ymin, bb.zmax-bb.zmin)
        if ext > 1e6:
            vertices, triangles = tessellate_solid(as_shape(shape).wrapped, n_points=1000)
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            mins, maxs = mesh.bounds
            xmin, ymin, zmin = map(float, mins)
            xmax, ymax, zmax = map(float, maxs)
            bbox_min = min(xmin, ymin, zmin)
            bbox_max = max(xmax, ymax, zmax)
            ext = max(xmax-xmin, ymax-ymin, zmax-zmin)
            flag = False
            penalty = validate(shape, mesh, flag)

            if penalty == 0:
                return flag, mesh, defaults

            if penalty < best_penalty:
                best_penalty = penalty
                best_result = (flag, mesh.copy(), defaults.copy())

        else:

            bbox_min = min(bb.xmin,bb.ymin, bb.zmin)
            bbox_max = max(bb.xmax,bb.ymax, bb.zmax)
            penalty = validate(shape)
            
            if penalty == 0:
                return defaults

            if penalty < best_penalty:
                best_penalty = penalty
                best_result = defaults.copy()

        if bbox_min < -100 or bbox_max > 100:
            factor = factor_scale_corner / max(abs(bbox_min), bbox_max)
            factor_scale_corner = update_factor_scale(factor_scale_corner, key='corner')
        elif ext < 100:
            factor = factor_scale_ext / ext
            factor_scale_ext = update_factor_scale(factor_scale_ext)
        else:
            factor = 1

        if factor != 1:
            for p, v in defaults.items():
                if isinstance(v, float) and not p.startswith('n_') and p.lower() != 'n' and not any(sub in p.lower() for sub in ("angle",
                                                                                 "theta",
                                                                                 "phi",
                                                                                 "fraction", 
                                                                                 "deg",
                                                                                 "degree",
                                                                                 "frac",
                                                                                 "latitude",
                                                                                 "longtitude",
                                                                                 "ratio")):
                    prec = orig_prec[p]
                    if p.startswith('fillet_') and factor > 1:
                        continue
                    defaults[p] = round(defaults[p] * factor, prec)
                
    return best_result


def samples_for(generator, N=15, eps=0.4):
    defaults_all = {k: p.default for k, p in signature(generator).parameters.items()}
    filtered = [(k, v) for k, v in defaults_all.items() if classify(k, v) != "str"]
    
    shape = generator
    try:
        validation_penalty = validate(shape())
    except Exception as e:
        pass

    if validation_penalty != 0:
        try:
            defaults_all = {k: p.default for k, p in signature(shape).parameters.items()}
            defs = defaults_all.copy()
            out = rescale_shape(shape, defaults_all)
            if isinstance(out, tuple) and len(out) == 3:
                flag, mesh, defaults = out
                rescaled_result = shape(**defaults)
                validation_penalty = validate(rescaled_result, mesh=mesh, flag=flag)
            else:
                defaults = out
                rescaled_result = shape(**defaults)
                validation_penalty = validate(rescaled_result)
        except Exception as e:
            pass

    precisions = {k: get_precision(v) for k, v in defaults_all.items()}

    keys   = [k for k, _ in filtered]
    types  = [classify(k, v) for k, v in filtered]

    z0 = []
    for t, k in zip(types, keys):
        encoded = enc(t, defaults_all[k])
        z0.extend(encoded if isinstance(encoded, list) else [encoded])
    es         = cma.CMAEvolutionStrategy(z0, 0.3, {'popsize': 8+int(3*math.log(len(z0))), 'verbose': -9})
    archive_z  = []
    
    attempt = 0
    max_attempts = 100
    
    while len(archive_z) < N and attempt < max_attempts:
        attempt += 1
        
        Zpop   = es.ask()
        fit    = []
        eps = max(0.05, 0.4 * (1 - attempt/max_attempts))

        for z in Zpop:
            params = decode_z(z, types, keys, precisions)
            for k, v in defaults_all.items():
                if k not in params:
                    params[k] = v
            try:
                shape = generator(**params)
            except Exception as e:
                fit.append(1e6)
                continue
            f = validate(shape)

            if archive_z:
                dist = min(distance(z, za) for za in archive_z)
                if dist < eps:
                    f += 1e3*(eps - dist)
            fit.append(f)

            if f < 1e-6:
                archive_z.append(list(z))
                if len(archive_z) == N: break

        if len(archive_z) == N:
            break

        es.tell(Zpop, fit)

    return [decode_z(z, types, keys, precisions) for z in archive_z]


# ========== Part processing ==========


def process_part(part, num_samples=10, max_attempts=10):
    """
    Worker that handles one part:
    1. Define the CAD function from code.
    2. Generate and save the default script.
    3. Run sampling (isolate_sampling) to get valid parameter sets.
    4. For each valid set, create a script, exec it, compare shapes, and save if valid.
    5. Return the part name if it FAILED (to record later), or None if successful ({num_samples} scripts generated).
    """
    name = part["name"]
    code_str = part["code"]
    
    # 1. Define the function from code_str
    ns_func = {"cq": cq, "math": math}
    try:
        exec(code_str, ns_func)
    except Exception:
        return name
    func = ns_func.get(name)
    defaults = {k: p.default for k, p in signature(func).parameters.items()}
    if not callable(func):
        append_txt_line(LOG_PATH, f"{name}: Can't define function (stage 1)")
        return name
    
    # 2. Save the original script (default_script)
    ns_exec = {"cq": cq, "math": math, "random": random, "np": np, "Voronoi": Voronoi}
    # part_folder = SAVE_DIR / name
    # part_folder.mkdir(parents=True, exist_ok=True)
    try:
        default_script = create_script(code_str, defaults)
        if not default_script:
            append_txt_line(LOG_PATH, f"{name}: Can't create default script (stage 2)")
            return name
        # exec(default_script, ns_exec)
        # with open(part_folder / "default_script.py", "w") as f:
            # f.write(default_script)
    except Exception as e:
        append_txt_line(LOG_PATH, f"{name}: Can't create default script (stage 2): ({e})")
        return name
    
    found_count = 0
    attempts = 0

    # 3. Sampling parameters
    while found_count < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            valid_sets = samples_for(func, N=15)
            append_txt_line(LOG_PATH, f"{name}: valid sets: {valid_sets} ")
            if not valid_sets:
                append_txt_line(LOG_PATH, f"{name}: 0 valid sets (stage 3)")
                return name
        except Exception as e:
            append_txt_line(LOG_PATH, f"{name}: Sampling failed (stage 3): ({e})")
            return name
        
        # 4. Loop through sampled parameter sets and validate
        for params in valid_sets:
            # 4a. Generate script for this parameter set
            try:
                script_code = create_script(code_str, params)
                if not script_code:
                    append_txt_line(LOG_PATH, f"{name}: can't create script code (stage 4a)")
                    continue
            except Exception:
                append_txt_line(LOG_PATH, f"{name}: can't create script code (stage 4a)")
                continue

            script_code = ensure_required_imports(script_code)

            # 4b. Execute the generated script to get its result
            try:
                exec(script_code, ns_exec)
            except Exception as e:
                append_txt_line(LOG_PATH, f"{name}: can't exec script code (stage 4b): {e}")
                continue
            if "result" not in ns_exec:
                append_txt_line(LOG_PATH, f"{name}: result not in ns_exec (stage 4b)")
                continue
            script_wp = ns_exec["result"]
            try:
                script_shape = script_wp.val().wrapped
            except Exception as e:
                append_txt_line(LOG_PATH, f"{name}: can't make wrapped model (stage 4b): {e}")
                continue
            
            script_code = collapse_script(script_code, ns_exec)
            script_code = remove_trailing_dot_zero(script_code)

            # 4c. Execute the original function with same params
            try:
                func_wp = func(**params)
            except Exception as e:
                append_txt_line(LOG_PATH, f"{name}: can't create (stage 4c): {e}")
                continue
            
            
            if "random" not in script_code:
                # 4d. Compare shapes via cuts: they should not differ
                try:
                    mesh_script = compound_to_mesh(script_wp.val())
                    mesh_func   = compound_to_mesh(func_wp.val())

                    if meshes_differ(mesh_script, mesh_func, tol=1e-6):
                        append_txt_line(LOG_PATH, f"{name}: mesh diff (stage 4d)")
                        continue
                except Exception as e:
                    append_txt_line(LOG_PATH, f"{name}: mesh check failed (stage 4d): {e}")
                    continue
            
            
            # 4e. Check CAD validity (c1, c2, c3)
            c1, c2, c3 = check_shape_validity(script_shape)
            if not (c1 and c2 and c3):
                append_txt_line(LOG_PATH, f"{name}: didn't pass the validitation (stage 4e)")
                continue
            
            # 4f. Save the script as one of the {num_samples} outputs
            save_script_to_file(name, found_count, script_code, num_samples)
            found_count += 1
            if found_count == num_samples:
                break

    if found_count < num_samples:
        append_txt_line(LOG_PATH, f"{name}: only {found_count}/{num_samples} valid scripts after {attempts} attempts")
        return name
    
    return None


def _run_child(conn, part, num_samples):
    try:
        res = process_part(part, num_samples)
        conn.send(res)
    finally:
        conn.close()


def timed_process_part(part, num_samples=10, timeout=300):
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_run_child, args=(child, part, num_samples))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        parent.close()
        return "__TIMEOUT__"

    result = parent.recv() if parent.poll() else "__CRASH__"
    parent.close()
    return result         


def init_worker():
    import os as _os
    _os.environ["OMP_NUM_THREADS"]       = "1"
    _os.environ["OPENBLAS_NUM_THREADS"]  = "1"
    _os.environ["MKL_NUM_THREADS"]       = "1"

    import json as _json
    import random as _random
    import math as _math
    import ast as _ast
    import re as _re
    import time as _time
    import traceback as _traceback

    import numpy as _np
    import cadquery as _cq
    import trimesh as _trimesh
    import cma as _cma

    from cadquery import Vector as _Vector, Plane as _Plane, Workplane as _Workplane, Shape as _Shape
    from scipy.spatial import Voronoi as _Voronoi
    from scipy.spatial import cKDTree as _cKDTree
    from pathlib import Path as _Path
    from inspect import signature as _signature

    from OCP.TopAbs import TopAbs_SOLID as _TopAbs_SOLID, TopAbs_SHELL as _TopAbs_SHELL, TopAbs_COMPOUND as _TopAbs_COMPOUND
    from OCP.TopAbs import TopAbs_Orientation as _TopAbs_Orientation, TopAbs_FACE as _TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer as _TopExp_Explorer
    from OCP.TopoDS import TopoDS_Shape as _TopoDS_Shape, TopoDS_Solid as _TopoDS_Solid, TopoDS as _TopoDS
    from OCP.TopLoc import TopLoc_Location as _TopLoc_Location
    from OCP.BRep import BRep_Tool as _BRep_Tool
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape as _BRepExtrema_DistShapeShape
    from OCP.BRepMesh import BRepMesh_IncrementalMesh as _BRepMesh_IncrementalMesh

    from multiprocessing import Process as _Process, Queue as _Queue, get_context as _get_context, TimeoutError as _TimeoutError
    from multiprocessing.pool import Pool as _Pool
    from functools import partial as _partial
    from tqdm import tqdm as _tqdm

    _g = globals()
    _g.update({
        'os': _os,
        'json': _json,
        'random': _random,
        'math': _math,
        'ast': _ast,
        're': _re,
        'time': _time,
        'traceback': _traceback,
        'np': _np,
        'cq': _cq,
        'trimesh': _trimesh,
        'cma': _cma,
        'Vector': _Vector,
        'Plane': _Plane,
        'Workplane': _Workplane,
        'Shape': _Shape,
        'Voronoi': _Voronoi,
        'Path': _Path,
        'signature': _signature,
        'TopAbs_SOLID': _TopAbs_SOLID,
        'TopAbs_SHELL': _TopAbs_SHELL,
        'TopAbs_COMPOUND': _TopAbs_COMPOUND,
        'TopAbs_Orientation': _TopAbs_Orientation,
        'TopAbs_FACE': _TopAbs_FACE,
        'TopExp_Explorer': _TopExp_Explorer,
        'TopoDS_Shape': _TopoDS_Shape,
        'TopoDS_Solid': _TopoDS_Solid,
        'TopoDS': _TopoDS,
        'TopLoc_Location': _TopLoc_Location,
        'BRep_Tool': _BRep_Tool,
        'BRepExtrema_DistShapeShape': _BRepExtrema_DistShapeShape,
        'BRepMesh_IncrementalMesh': _BRepMesh_IncrementalMesh,
        'Process': _Process,
        'Queue': _Queue,
        'get_context': _get_context,
        'TimeoutError': _TimeoutError,
        'Pool': _Pool,
        'partial': _partial,
        'tqdm': _tqdm,
    })


# ========== Pipeline Execution ==========


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--code-db", type=str, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--n-workers", type=int, required=True)
    parser.add_argument("--task-timeout", type=int, required=True)
    args = parser.parse_args()

    LOG_PATH = args.log_path
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_DIR = args.save_dir
    NUM_SAMPLES = args.num_samples
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    N_WORKERS = args.n_workers
    TASK_TIMEOUT = args.task_timeout
    CODE_DB = args.code_db

    db = load_code_db(CODE_DB)
    print("All parts:", len(db))

    width = max(2, len(str(NUM_SAMPLES)))
    done_file = f"{NUM_SAMPLES:0{width}d}.py"
    already_generated_names = {p.name for p in SAVE_DIR.iterdir() if p.is_dir() and (p / done_file).is_file()}

    print("Generated parts", len(already_generated_names))
    valid_parts_array = [item for item in db if item['name'] not in already_generated_names]
 
    parts = valid_parts_array

    POOL = None
    if POOL is None:
        from multiprocessing import get_context
        ctx = get_context("forkserver")
        POOL = NonDaemonPool(
            processes=N_WORKERS,
            initializer=init_worker,
            context=ctx
        )
        print("POOL Initialized", flush=True)

        import atexit
        atexit.register(lambda: (POOL.close(), POOL.join()))

    async_results = [
                        (part['name'],
                        POOL.apply_async(timed_process_part, args=(part, NUM_SAMPLES)))
                        for part in parts
                    ]

    for name, ar in async_results:
        result = ar.get()
        if result == "__TIMEOUT__":
            append_txt_line(LOG_PATH, f"{name}: TASK TIMEOUT ({TASK_TIMEOUT}s)")
            continue
        if result == "__CRASH__":
            append_txt_line(LOG_PATH, f"{name}: TASK CRASHED")
            continue
        if result is not None:
            append_txt_line(LOG_PATH, result)