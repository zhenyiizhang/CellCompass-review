"""Seal or verify the compact numerical input bundle."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal',action='store_true',help='Maintainer: record current input hashes')
    args=parser.parse_args()
    manifest=ROOT/'metadata/plot_data_manifest.json'
    if args.seal:
        rows=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((ROOT/'plot_data').rglob('*')) if p.is_file()]
        manifest.write_text(json.dumps(rows,indent=2)+'\n')
    rows=json.loads(manifest.read_text())
    for row in rows:
        p=ROOT/row['path']
        assert p.is_file(),row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
    print(f'Verified {len(rows)} compact plotting inputs')

if __name__=='__main__':main()
