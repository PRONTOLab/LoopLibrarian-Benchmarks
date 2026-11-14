#!/usr/bin/env python3
"""Run all 598 ATen fixtures or the 115-case paper timing and plotting cohort."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/artifact'), str(ROOT/'scripts/correctness')]
from common import OUT, PYTHON, command, save, setting


def select_cases(cohort, cases=None):
    checklist = ROOT/'aten'/('all_cases.csv' if cohort == 'all' else 'paper_cases.csv')
    with checklist.open() as stream:
        rows = list(csv.DictReader(stream))
    names = [r['kernel'] for r in rows]
    wanted = {n if n.startswith('aten_') else 'aten_'+n for n in cases} if cases else set(names)
    if wanted - set(names):
        raise ValueError('Cases outside the '+cohort+' cohort: '+', '.join(sorted(wanted-set(names))))
    return checklist, [r for r in rows if r['kernel'] in wanted]


def record_selection(args, out):
    path = out/'selection.json'
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous and previous['cohort'] != args.cohort:
        raise ValueError('Output belongs to another ATen cohort; use a separate directory')
    cases = args.cases or (previous['cases'] if previous else None)
    provenance = out/'matching/provenance.json'
    if provenance.exists():
        old = json.loads(provenance.read_text())
        shapes = old['arguments'].get('shape_specs')
        if bool(shapes) != (args.cohort == 'paper'):
            raise ValueError('Saved matching dimensions belong to another cohort')
        if shapes and old.get('hashes', {}).get(shapes) != hashlib.sha256((ROOT/'aten/paper_shapes.json').read_bytes()).hexdigest():
            raise ValueError('Saved shape manifest differs from the paper dimensions')
        if cases is None:
            cases = old['selected']
        normalized = {n if n.startswith('aten_') else 'aten_'+n for n in cases}
        if normalized != set(old['selected']):
            raise ValueError('Resume must keep the saved case selection')
    checklist, rows = select_cases(args.cohort, cases)
    hashes = {r['kernel']: hashlib.sha256((ROOT/r['source']).read_bytes()).hexdigest() for r in rows}
    for r in rows:
        if r.get('source_sha256') and hashes[r['kernel']] != r['source_sha256']:
            raise ValueError('Inventory source changed: '+r['kernel'])
    selection = dict(cohort=args.cohort, cases=[r['kernel'] for r in rows],
                     source_sha256=hashes, dimensions='paper' if args.cohort == 'paper' else 'original fixtures',
                     numeric_model=args.numeric_model if args.cohort == 'all' else 'recorded paper recipes',
                     timing_correctness=args.timing_correctness)
    if previous and previous != selection:
        raise ValueError('Saved selection, source hashes, or numerical policy changed; use a fresh output')
    out.mkdir(parents=True, exist_ok=True)
    save(path, selection)
    return checklist, selection['cases']


def all_matching_options(names, numeric_model):
    paper = {r['kernel']: r.get('match_options', {}) for r in
             json.loads((ROOT/'aten/matching_options.json').read_text())}
    rows = []
    for name in names:
        options = dict(ranks=[1], region=True, split_pointwise_inputs=True)
        options.update(paper.get(name, {}))
        # Reuse generic normalization settings without silently relaxing cases.
        options['numeric_model'] = numeric_model
        rows.append(dict(kernel=name, match_options=options))
    return rows


def paper_report(out, runtime):
    command([*runtime, '--stage', 'report'], out/'report.log')
    report = json.loads((out/'gpu/report.json').read_text())
    save(out/'summary.json', dict(**report['summary'], cohort='paper',
        scope='Paper dimensions and recorded recipes; incomplete matches and failed comparisons remain explicit.',
        details='gpu/report.json'))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', choices=['all', 'paper'], default='paper')
    p.add_argument('--stage', choices=['match','build','run','report','summarize','plot','all'], default='all')
    p.add_argument('--cases', nargs='+', help='Explicit subset of the selected cohort')
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--numeric-model', choices=['strict','relaxed'],
                   help='Full-suite EqSat model; paper mode retains recorded per-case models')
    p.add_argument('--timing-correctness', choices=['strict','publication'])
    p.add_argument('--paper-figure', type=Path, help='Optional original PDF for the paper plot comparison')
    args = p.parse_args(argv)
    args.stage = 'report' if args.stage == 'summarize' else args.stage
    if args.cohort == 'paper' and args.numeric_model is not None:
        p.error('Paper mode uses the numerical models in aten/matching_options.json')
    out = args.output.resolve()
    previous_path = out/'selection.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    args.numeric_model = args.numeric_model or previous.get('numeric_model', 'strict')
    args.timing_correctness = args.timing_correctness or previous.get('timing_correctness', 'strict')
    if args.jobs < 1:
        p.error('--jobs must be positive')
    if args.cohort == 'all' and (args.stage == 'plot' or args.paper_figure or args.timing_correctness != 'strict'):
        p.error('Paper plotting and publication timing criteria require --cohort paper')
    campaign = out/'matching'
    fresh = args.stage in ('match', 'all')
    if fresh and any(p.exists() for p in (campaign, out/'selection.json', out/'gpu')):
        p.error('Use a fresh output for match/all; build/run/report/plot resume saved results')
    if args.stage in ('run','report','plot') and not (campaign/'provenance.json').exists():
        p.error('No saved matching campaign; run --stage match or build first')
    try:
        checklist, names = record_selection(args, out)
    except ValueError as error:
        p.error(str(error))
    selection = ['--cases', *names]
    match_needed = fresh or (args.stage == 'build' and not (campaign/'provenance.json').exists())

    if args.cohort == 'all':
        options = out/'matching-options.json'
        if match_needed:
            save(options, all_matching_options(names, args.numeric_model))
        base = [PYTHON, ROOT/'aten/pointwise.py', '--scope', 'all', '--output', out,
                '--checklist', checklist, *selection, '--jobs', str(args.jobs),
                '--numeric-model', args.numeric_model]
        stages = ['match'] if match_needed else []
        if args.stage in ('all','build'):
            stages.append('build')
        if args.stage in ('all','run'):
            stages.append('run')
        if args.stage == 'report':
            stages.append('report')
        for stage in stages:
            extra = ['--case-options', options] if stage == 'match' else []
            command([*base, '--stage', stage, *extra], out/('driver-'+stage+'.log'), timeout=21600)
        summary_path = out/'summary.json'
        if summary_path.exists():
            print(json.dumps(json.loads(summary_path.read_text())['summary']), flush=True)
        return

    runtime = [PYTHON, ROOT/'scripts/correctness/aten_paper_runtime.py',
               '--campaign', campaign, '--output', out/'gpu', *selection,
               '--cudnn-plan-cache','--gpu-workspace-cache','--persistent-workspace',
               '--native-paper-inputs','--timing-correctness',args.timing_correctness]
    if match_needed:
        command([PYTHON, ROOT/'scripts/correctness/aten_paper115_campaign.py',
            '--output', campaign, '--checklist', checklist, *selection, '--include-validated',
            '--shape-specs', ROOT/'aten/paper_shapes.json',
            '--case-options', ROOT/'aten/matching_options.json', '--jobs', str(args.jobs)],
            out/'match.log', timeout=21600)
    if args.stage in ('build','all'):
        command([*runtime,'--stage','build','--jobs',str(args.jobs)], out/'build.log', timeout=21600)
    if args.stage in ('run','all'):
        command([*runtime,'--stage','run','--transport', ROOT/'scripts/artifact/remote.py',
                 '--remote-library-path', setting('remote_library_path',
                    '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu')], out/'run.log', timeout=21600)
    report = paper_report(out, runtime)
    if args.stage in ('plot','all'):
        if report['summary']['timing_pairs_validated']:
            extra = ['--paper-figure', args.paper_figure.resolve()] if args.paper_figure else []
            command([PYTHON, ROOT/'aten/plot_results.py', '--report', out/'gpu/report.json',
                     '--output', out/'plots', *extra], out/'plot.log')
        elif args.stage == 'plot':
            p.error('No validated paper-cohort timings are available to plot')
        else:
            save(out/'plots/status.json', dict(status='not_plotted', reason='No validated timing pairs'))
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
