#!/usr/bin/env python3
"""Fresh paper-cohort raising, proof, GPU correctness and five-pair timings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/artifact'), str(ROOT/'scripts/correctness')]
from common import OUT, PYTHON, command, save, setting


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['build','run','summarize','all'],default='all')
    p.add_argument('--cases',nargs='+',help='Explicit subset; omit for all 115 paper cases')
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--jobs',type=int,default=3)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    campaign=out/'matching';gpu=out/'gpu'
    selection=['--cases',*args.cases] if args.cases else []
    runtime=[PYTHON,ROOT/'scripts/correctness/aten_paper_runtime.py',
             '--campaign',campaign,'--output',gpu,*selection,
             '--cudnn-plan-cache','--gpu-workspace-cache','--persistent-workspace',
             '--native-paper-inputs']
    if args.stage in ('build','all'):
        command([PYTHON,ROOT/'scripts/correctness/aten_paper115_campaign.py',
            '--output',campaign,*selection,'--include-validated',
            '--shape-specs',ROOT/'aten/paper_shapes.json',
            '--case-options',ROOT/'aten/matching_options.json','--jobs',str(args.jobs)],
            out/'match.log',timeout=21600)
        command([*runtime,'--stage','build','--jobs',str(args.jobs)],
                out/'build.log',timeout=21600)
    if args.stage in ('run','all'):
        command([*runtime,'--stage','run','--transport',ROOT/'scripts/artifact/remote.py',
                 '--remote-library-path',setting('remote_library_path',
                    '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu')],
                out/'run.log',timeout=21600)
    command([*runtime,'--stage','report'],out/'report.log')
    report=json.loads((gpu/'report.json').read_text())
    save(out/'summary.json',dict(**report['summary'],
        scope='Paper dimensions and recorded recipes; incomplete matches and failed comparisons remain explicit.',
        details='gpu/report.json'))
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
