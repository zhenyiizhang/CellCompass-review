"""Run every figure with original workstation paths blocked, then inspect exports.

This is a reproducibility diagnostic, not a security sandbox. Fonts and Python
packages outside the checkout remain accessible. Study-data access must not use
the original project, agent home or another copy of this figure repository.
This checkout is exempt from the forbidden prefixes, so the test still runs when
it is cloned inside one of them. Any other copy there remains rejected.
"""
from pathlib import Path
import argparse
import json
import os
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN=[
    '/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309',
    '/Users/zhenyizhang/.cellcompass',
    '/Users/zhenyizhang/Desktop',
]

def under(parent,path):
    return path==parent or path.startswith(parent.rstrip('/')+'/')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forbid',action='append',default=[])
    args=parser.parse_args()
    forbidden=FORBIDDEN+args.forbid
    checkout={str(ROOT),os.path.realpath(ROOT)}
    def audit(event,values):
        if event not in ('open','os.listdir','os.scandir'):return
        raw=values[0]
        if not isinstance(raw,(str,bytes,os.PathLike)):return
        path=os.path.abspath(os.fsdecode(raw))
        if any(under(base,path) for base in checkout):return
        if any(under(p,path) for p in forbidden):
            raise RuntimeError(f'External study path accessed: {path}')
    sys.addaudithook(audit)
    sys.path.insert(0,str(ROOT))
    sys.argv=['reproduce.build','--output',str(ROOT/'output/clean-reproduction')]
    runpy.run_module('reproduce.build',run_name='__main__')
    import pymupdf
    output=ROOT/'output/clean-reproduction'
    files=sorted(output.glob('*Figure_*.pdf'))
    assert len(files)==20, len(files)
    for path in files:
        with pymupdf.open(path) as doc:
            assert len(doc)==1
            assert len(doc[0].get_drawings()) or len(doc[0].get_images())
    (output/'independent_path_check.json').write_text(json.dumps({
        'status':'passed','figures':20,'forbidden_paths':forbidden,'checkout':str(ROOT),
        'scope':'Redraw from bundled final analysis data; no model fitting.'
    },indent=2)+'\n')
    print('PASS: all 20 figures rebuilt without external study paths')

if __name__=='__main__':main()
