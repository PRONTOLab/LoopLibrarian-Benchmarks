#!/usr/bin/env python3
"""Regenerate source specification counts and compare with the reported total."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/artifact'), str(ROOT/'scripts/correctness')]
import argparse,json,collections
from common import COMPILER, OUT, save
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--stage',choices=['build','run','summarize','all'],default='all');p.parse_args()
rows=[]
for path in sorted((COMPILER/'generic_solver/lib_v3/manifest').glob('*.json')):
    fragment=json.loads(path.read_text())
    for spec in fragment['specs']:
        rows.append(dict(name=spec['name'],library=spec.get('target',{}).get('library'),manifest=str(path.relative_to(COMPILER)),source=fragment.get('source')))
counts=collections.Counter(r['library'] for r in rows)
save(OUT/'inventory.json',dict(total=len(rows),reported_total=257,difference=len(rows)-257,by_library=dict(counts),specifications=rows))
print(json.dumps(dict(total=len(rows),reported_total=257,by_library=dict(counts)),indent=2))
