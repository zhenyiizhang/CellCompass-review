"""Rebuild the current paper's 6 main and 14 SI figures from bundled inputs."""
from pathlib import Path
import argparse
import json
import os
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figure_reproduction'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/review_figures')
    args=parser.parse_args()
    out=args.output.resolve()
    expected=[f'Main_Figure_{i:02}' for i in range(1,7)]+[f'Supplementary_Figure_{i:02}' for i in range(1,15)]
    if any((out/(stem+'.pdf')).exists() for stem in expected):
        parser.error('Use a new output directory so stale figures cannot pass verification.')
    forbidden=[
        '/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309',
        '/Users/zhenyizhang/Documents/GitHub/CellCompass-figure-code',
        '/Users/zhenyizhang/.cellcompass','/Users/zhenyizhang/Desktop',
    ]
    def under(base,p):return p==base or p.startswith(base.rstrip('/')+'/')
    def audit(event,values):
        if event not in {'open','os.listdir','os.scandir'}:return
        raw=values[0]
        if not isinstance(raw,(str,bytes,os.PathLike)):return
        path=os.path.realpath(os.path.abspath(os.fsdecode(raw)))
        if under(str(ROOT),path) or under(str(out),path):return
        if any(under(p,path) for p in forbidden):raise RuntimeError(f'Original study path requested: {path}')
    sys.addaudithook(audit)
    sys.path.insert(0,str(FIG))
    started=time.perf_counter()
    sys.argv=['reproduce.build','--only','si1','emt','pulse','lineage','protein','protein-distribution','main','--output',str(out)]
    runpy.run_module('reproduce.build',run_name='__main__')
    import pymupdf
    for stem in expected:
        for suffix in ['.pdf','.png']:
            assert (out/(stem+suffix)).stat().st_size>1000,stem+suffix
        with pymupdf.open(out/(stem+'.pdf')) as doc:
            assert len(doc)==1,stem
            assert doc[0].get_drawings() or doc[0].get_images(),stem
    report={'status':'passed','figures':len(expected),'elapsed_seconds':round(time.perf_counter()-started,2),
            'scope':'Redraw from bundled final analysis data, without model refitting or LLM access.'}
    (out/'review_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
