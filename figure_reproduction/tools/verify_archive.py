"""Validate archived file hashes, figure links and Python syntax. No analysis runs."""
from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
errors = []
rows = list(csv.DictReader((ROOT/'metadata/file_manifest.csv').open()))
for row in rows:
    path = ROOT/row['path']
    if not path.is_file():
        errors.append(f'Missing: {row["path"]}')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        errors.append(f'Checksum mismatch: {row["path"]}')
scripts = list((ROOT/'source').rglob('*.py'))
for path in scripts:
    try:
        ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        errors.append(str(exc))
figures = json.loads((ROOT/'metadata/figure_map.json').read_text())
for fig in figures:
    for script in fig['scripts']:
        if not (ROOT/'source'/script).is_file():
            errors.append(f'{fig["figure"]}: missing script {script}')
print(json.dumps(dict(files=len(rows), python_scripts=len(scripts),
                     figures=len(figures), errors=errors), indent=2))
raise SystemExit(bool(errors))
