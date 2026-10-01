"""Verify the source/data snapshot without importing optional scientific libraries."""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    'GitHub token':rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    'GitHub fine-grained token':rb'\bgithub_pat_[A-Za-z0-9_]{40,}\b',
    'Overleaf token':rb'\bolp_[A-Za-z0-9]{25,}\b',
    'API secret':rb'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{30,}\b',
    'Private key':rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
}

def main():
    rows=json.loads((ROOT/'metadata/review_manifest.json').read_text())
    errors=[]
    for row in rows:
        p=ROOT/row['path']
        if not p.is_file():
            errors.append(f'Missing {row["path"]}');continue
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            errors.append(f'Hash mismatch {row["path"]}')
        if p.suffix=='.py':
            try:ast.parse(data.decode('utf-8-sig'),filename=str(p))
            except (SyntaxError,UnicodeError) as e:errors.append(f'{row["path"]}: {e}')
        if p.suffix in {'.py','.json','.yaml','.yml','.md','.txt','.js','.jsx','.mjs','.html','.csv','.tsv'}:
            for name,pattern in PATTERNS.items():
                if re.search(pattern,data):errors.append(f'{name} pattern in {row["path"]}')
    print(json.dumps({'files':len(rows),'errors':errors},indent=2))
    raise SystemExit(bool(errors))

if __name__=='__main__':main()
