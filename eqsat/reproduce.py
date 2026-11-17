"""Rebuttal coverage: all 30 inputs, both EqSat arms, and real SpEQ."""
import argparse
import csv
import json
from pathlib import Path
from common import ROOT, OUT, INPUTS, save
from coverage import campaign
import speq

PUBLISHED_COUNTS = dict(inputs=30, full=17, none=8, speq_linearized_gemm=1)

def submitted_counts_match(totals):
    # Both LoopLibrarian counts cover the same complete 30-program inventory.
    return (totals['full']['recognized']==PUBLISHED_COUNTS['full'] and
            totals['none']['recognized']==PUBLISHED_COUNTS['none'] and
            totals['speq']['linearized_gemm_matches']==PUBLISHED_COUNTS['speq_linearized_gemm'])

def inputs():
    rows = []
    for row in csv.DictReader((INPUTS/'polybench_manifest.csv').open()):
        src = ROOT/row['source']
        rows.append(dict(name=row['kernel'], source=str(src),
            function='kernel_'+row['kernel'].replace('-', '_'),
            flags=['-I'+str(ROOT/'polybench/sources/utilities'),
                   '-DMINI_DATASET', '-DDATA_TYPE_IS_DOUBLE']))
    return rows

def summarize(rows):
    results = [json.loads((OUT/'ll'/r['name']/arm/'result.json').read_text())
               for r in rows for arm in ['full', 'none']]
    totals = {}
    for arm in ['full', 'none']:
        group = [r for r in results if r['ablation']==arm]
        totals[arm] = dict(inputs=len(group), recognized=sum(r['recognized'] for r in group),
            abi_lowered=sum(r['abi_ok'] for r in group),
            recognized_names=[r['name'] for r in group if r['recognized']],
            errors=[dict(name=r['name'], error=r['error']) for r in group if r['error']])
    native = json.loads((OUT/'speq-native/results.json').read_text())
    linear = json.loads((OUT/'speq-linearized/results.json').read_text())
    totals['speq'] = dict(unmodified_cases=len(native), matched_unmodified=sum(r.get('n_match',0)>0 for r in native),
                         linearized_gemm_matches=sum(r.get('n_match',0)>0 for r in linear))
    full = len(rows)==PUBLISHED_COUNTS['inputs']
    reproduced = submitted_counts_match(totals)
    save(OUT/'summary.json', dict(totals=totals, results=results, selected_inputs=len(rows), full_cohort=full,
         published_counts=PUBLISHED_COUNTS, submitted_counts_reproduced=full and reproduced,
         scope='Programs with at least one proven library replacement, across all selected PolyBench inputs; '
               'complete replacement is not required. ABI lowering is reported separately. '
               'SpEQ one is a linearized GEMM, not unmodified PolyBench.'))
    print(json.dumps(totals, indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['build','run','summarize','all'], default='all')
    p.add_argument('--cases', nargs='+'); a=p.parse_args()
    rows=[r for r in inputs() if not a.cases or r['name'] in a.cases]
    if not rows or (a.cases and set(a.cases)-{r['name'] for r in rows}): p.error('unknown PolyBench case')
    if a.stage in ('build','all'):
        save(OUT/'cases.json', rows)
        campaign(rows, OUT/'ll', ablations=('full','none'))
        base=ROOT/'polybench/sources'
        speq.run(OUT/'speq-native', [str(Path(r['source']).resolve()) for r in rows],
                 func='AUTO', cflags=f'-I"{base}/utilities" -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE')
        speq.run(OUT/'speq-linearized', [speq.source_path('benchmarks/gemm_ref.c')], func='gemm_ref')
    if a.stage in ('run','summarize'):
        rows=json.loads((OUT/'cases.json').read_text())
    summarize(rows)

if __name__=='__main__': main()
