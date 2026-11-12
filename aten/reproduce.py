#!/usr/bin/env python3
"""Fresh ATen raising/matching and paper-shape execution. Partial cohort remains explicit."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/artifact'), str(ROOT/'scripts/correctness')]
import argparse,json
from common import OUT, PYTHON, command, setting, save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['build','run','summarize','all'],default='all')
    p.add_argument('--cases',nargs='+',required=True,help='Explicit cases; pending cases are not silently skipped')
    p.add_argument('--matching-stage',choices=['debuf','linalg'],default='debuf')
    p.add_argument('--ranks',nargs='+',default=['0'])
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    shapes=ROOT/'aten/paper_shapes.json';campaign=OUT/'matching';gpu=OUT/'gpu'
    if a.stage in ('build','all'):
        command([PYTHON,ROOT/'scripts/correctness/aten_paper115_campaign.py','--output',campaign,
                 '--cases',*a.cases,'--include-validated','--shape-specs',shapes,'--region',
                 '--region-budget','120','--numeric-model','relaxed','--matching-stage',a.matching_stage,
                 '--ranks',*a.ranks],OUT/'match.log',timeout=7200)
        command([PYTHON,ROOT/'scripts/correctness/aten_fixture_batch.py','--campaign',campaign,
                 '--output',gpu,'--cases',*a.cases,'--shape-specs',shapes,'--rank','-1',
                 '--persistent-workspace','--timing-samples','5','--stage','build'],OUT/'build.log',timeout=14400)
    if a.stage in ('run','all'):
        command([PYTHON,ROOT/'scripts/correctness/aten_fixture_batch.py','--campaign',campaign,
                 '--output',gpu,'--cases',*a.cases,'--shape-specs',shapes,'--rank','-1',
                 '--timing-samples','5','--stage','run','--correctness-kinds','paper',
                 '--cudnn-plan-cache','--transport',ROOT/'scripts/artifact/remote.py'],OUT/'run.log',timeout=14400)
    results=[json.loads(p.read_text()) for p in gpu.glob('*/gpu-result.json')]
    save(OUT/'summary.json',dict(cases=a.cases,results=results,scope='Explicit subset; full 115-case reproduction remains incomplete'))

if __name__=='__main__':main()
