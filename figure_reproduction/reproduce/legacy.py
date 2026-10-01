"""Load drawing functions without executing study analysis or training entry points.

The source snapshot remains immutable. Only imports, declarations, constants and
Matplotlib style settings are evaluated. Model/agent imports are deliberately
excluded. Drawing functions consume the frozen plot tables supplied by build.py.
"""
from __future__ import annotations
import ast
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
SAFE={'__future__','pathlib','typing','collections','dataclasses','math','json','re',
      'os','sys','argparse','itertools','textwrap','numpy','pandas','matplotlib','mpl_toolkits',
      'scipy','sklearn','seaborn','PIL','zipfile','types','importlib','base64','html'}

def load(relative: str, **overrides):
    path=ROOT/'source'/relative
    tree=ast.parse(path.read_text())
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    for node in tree.body:
        if isinstance(node,ast.Import):
            names=[n for n in node.names if n.name.split('.')[0] in SAFE]
            if names:nodes.append(ast.Import(names=names))
        elif isinstance(node,ast.ImportFrom):
            if node.module and node.module.split('.')[0] in SAFE:nodes.append(node)
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            # Decorated model functions are not used by the plotting entry points.
            node.decorator_list=[]
            nodes.append(node)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            nodes.append(node)
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            text=ast.unparse(node.value.func)
            if text.endswith('rcParams.update'):nodes.append(node)
    module=types.ModuleType(path.stem)
    module.__file__=str(path)
    env=module.__dict__
    # Evaluate each top-level declaration individually so optional model constants
    # cannot force a model import. Any unresolved constant required by a drawing
    # function still raises an ordinary NameError when that function is called.
    deferred=[]
    for node in nodes:
        unit=ast.fix_missing_locations(ast.Module(body=[nodes[0],node],type_ignores=[]))
        try:exec(compile(unit,str(path),'exec'),env)
        except (NameError,ModuleNotFoundError) as exc:
            deferred.append((ast.unparse(node),str(exc)))
    # Remap original workspace constants recursively. No original workstation
    # path is used for input or output by the reproducible entry points.
    original='/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309'
    def remap(value):
        if isinstance(value,Path) and str(value).startswith(original):
            return ROOT/'source'/str(value)[len(original)+1:]
        if isinstance(value,dict):return {k:remap(v) for k,v in value.items()}
        if isinstance(value,list):return [remap(v) for v in value]
        return value
    for key,value in list(env.items()):
        if key.isupper():env[key]=remap(value)
    env.update(overrides)
    env['_skipped_optional_declarations']=deferred
    return module
