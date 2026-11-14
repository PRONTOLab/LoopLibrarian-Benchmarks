#!/usr/bin/env python3
"""Reproduce pointwise candidate matching, strict GPU checks, and five pairs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/artifact'), str(ROOT/'scripts/correctness')]
from common import command, environment, save, setting
from aten_fixture_batch import NATIVE
from aten_paper115_campaign import complete_variant
from ll_paths import COMPILER


def report(out, selected, shape_specs=None, extra_gpu_results=(), scope='pointwise'):
    gpu_file = out/'gpu/gpu-results.json'
    gpu = {r['kernel']: r for r in json.loads(gpu_file.read_text())} if gpu_file.exists() else {}
    for path in extra_gpu_results:
        path = Path(path).resolve()
        for row in json.loads(path.read_text()):
            name = row['kernel']
            if name not in selected:
                continue
            if name in gpu:
                raise ValueError('Duplicate GPU evidence; select one run explicitly: '+name)
            built = json.loads((path.parent/name/'build.json').read_text())
            current = json.loads((out/'matching'/name/'result.json').read_text())
            if (built['source_sha256'] != current['source_sha256'] or
                    built['binary_sha256'] != row['binary_sha256']):
                raise ValueError('GPU evidence has a different source or binary: '+name)
            rank = 'rank'+str(built['rank'])
            measured_proof = json.loads((path.parent.parent/'matching'/name/rank/'proof.json').read_text())
            current_proof = json.loads((out/'matching'/name/rank/'proof.json').read_text())
            for field in ('library_sha256','semantic_inputs','configuration','abi_sha256'):
                if measured_proof[field] != current_proof[field]:
                    raise ValueError('GPU evidence uses different compiler/specification inputs: '+name)
            gpu[name] = dict(row, evidence_file=str(path))
    rows = []
    for name in selected:
        path = out/'matching'/name/'result.json'
        matched = json.loads(path.read_text()) if path.exists() else {}
        request_path = out/'matching'/name/'request.json'
        request = json.loads(request_path.read_text()) if request_path.exists() else {}
        previous = request.get('baseline_complete_match')
        if previous in ('True','False'):
            previous = previous == 'True'
        if not isinstance(previous,bool):
            previous = None
        variants = [v for v in matched.get('variants', []) if complete_variant(v)]
        considered = [v for v in matched.get('variants', []) if 'counts' in v]
        best = min(considered, key=lambda v: (v['counts']['linalg'],v['counts']['loops'])) if considered else {}
        runtime = gpu.get(name, {})
        build_root = Path(runtime['evidence_file']).parent if runtime.get('evidence_file') else out/'gpu'
        build_path = build_root/name/'build.json'
        built = json.loads(build_path.read_text()) if build_path.exists() else None
        native = name.removeprefix('aten_') in NATIVE
        same_binary = bool(built and built.get('binary_sha256') and
                           built['binary_sha256'] == runtime.get('binary_sha256') and
                           built.get('source_sha256') and
                           built['source_sha256'] == matched.get('source_sha256'))
        execution = ('incomplete_match' if not variants else
                     'native_adapter_unavailable' if not native else
                     'not_built' if built is None else
                     'build_failed' if built.get('rc') != 0 else
                     'not_run' if not runtime else
                     'stale_or_incomplete_gpu_evidence' if not same_binary else
                     'validated' if runtime.get('fully_validated') else 'correctness_failed')
        rows.append(dict(kernel=name, complete_match=bool(variants),
                         baseline_complete_match=previous,
                         numeric_model=request.get('numeric_model','unknown'),
                         match_status=matched.get('status', 'pending'),
                         emitted=best.get('emitted', []),
                         residual_counts=best.get('counts'),
                         matching_details='matching/'+name+'/result.json',
                         native_baseline_available=native, build=built, execution_status=execution,
                         strict_gpu_pass=bool(variants and built and built.get('rc') == 0 and
                                              same_binary and runtime.get('fully_validated', False)),
                         gpu=runtime or None))
    summary = dict(candidates=len(rows), complete_matches=sum(r['complete_match'] for r in rows),
                   strict_gpu_passes=sum(r['strict_gpu_pass'] for r in rows),
                   timed=sum(bool(r['strict_gpu_pass'] and r['gpu'].get('timing', {}).get('passed')
                                  and r['gpu']['timing'].get('native_over_raised')) for r in rows))
    if any(r['baseline_complete_match'] is not None for r in rows):
        summary['baseline_complete_matches'] = sum(r['baseline_complete_match'] is True for r in rows)
        summary['new_complete_matches'] = sum(r['complete_match'] and
            r['baseline_complete_match'] is False for r in rows)
    if scope == 'all':
        summary.update(native_adapters=sum(r['native_baseline_available'] for r in rows),
                       build_passes=sum(bool(r['build'] and r['build'].get('rc') == 0) for r in rows),
                       complete_without_native_adapter=sum(r['complete_match'] and
                           not r['native_baseline_available'] for r in rows))
    save(out/'summary.json', dict(summary=summary, cases=rows, cohort=scope,
         shape_specs=str(shape_specs) if shape_specs else None,
         scope=(('Full ATen fixture inventory, not the paper 115 timing cohort. ' if scope == 'all'
                 else 'Pointwise IR candidates, not the paper 115 denominator. ')+
                ('Explicit shape manifest; separate source copies. ' if shape_specs else 'Original fixture dimensions. ')+
                'Matching/ABI results are separate from optional GPU checks and timings.')))
    fields = ['kernel','numeric_model','baseline_complete_match','complete_match','strict_gpu_pass','timing_pairs',
              'native_over_raised','raised_min_ms','native_min_ms','dimension_scope',
              'native_baseline_scope','match_status','execution_status','build_error',
              'emitted','remaining_linalg','remaining_loops']
    with (out/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows:
            runtime = row['gpu'] or {}; timing = runtime.get('timing',{})
            timed = row['strict_gpu_pass'] and timing.get('passed')
            writer.writerow(dict(kernel=row['kernel'],numeric_model=row['numeric_model'],complete_match=row['complete_match'],
                baseline_complete_match=row['baseline_complete_match'],
                strict_gpu_pass=row['strict_gpu_pass'],
                timing_pairs=len(timing['samples'])//2 if timed else '',
                native_over_raised=timing.get('native_over_raised','') if timed else '',
                raised_min_ms=timing.get('minimum_ms',{}).get('raised','') if timed else '',
                native_min_ms=timing.get('minimum_ms',{}).get('native_aten_cuda','') if timed else '',
                dimension_scope=runtime.get('dimension_scope',''),
                native_baseline_scope=runtime.get('scope',''),match_status=row['match_status'],
                execution_status=row['execution_status'],build_error=(row['build'] or {}).get('error',''),
                emitted=';'.join(row['emitted']),
                remaining_linalg=(row['residual_counts'] or {}).get('linalg',''),
                remaining_loops=(row['residual_counts'] or {}).get('loops','')))
    print(json.dumps(summary), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['match','build','run','report','all'], default='all')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+')
    p.add_argument('--checklist',type=Path,default=ROOT/'aten/pointwise_candidates.csv',
                   help='Candidate inventory or a recorded subset, with source hashes')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--scope', choices=['pointwise','all'], default='pointwise',
                   help='Label the selected inventory; does not change the selected cases')
    p.add_argument('--case-options', type=Path, help='Saved generic matching options for the selected cases')
    p.add_argument('--ranks', type=int, nargs='+', default=[1],
                   help='Operand rank limits to test; default: 1')
    p.add_argument('--numeric-model',choices=['strict','relaxed'],default='strict',
                   help='EqSat assumptions; GPU output checks remain strict in either mode')
    p.add_argument('--shape-specs', type=Path,
                   help='Optional archived dimensions, applied to separate C copies')
    p.add_argument('--extra-gpu-results',type=Path,nargs='+',default=[],
                   help='Report only: combine disjoint runs with identical source/compiler/specification inputs')
    args = p.parse_args()
    if args.extra_gpu_results and args.stage != 'report':
        p.error('--extra-gpu-results is only supported with --stage report')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    checklist = args.checklist.resolve()
    records = list(csv.DictReader(checklist.open()))
    wanted = {n if n.startswith('aten_') else 'aten_'+n for n in args.cases} if args.cases else None
    if wanted and wanted - {r['kernel'] for r in records}:
        p.error('Unknown pointwise candidate')
    records = [r for r in records if wanted is None or r['kernel'] in wanted]
    for row in records:
        if hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest() != row['source_sha256']:
            p.error('Candidate source changed: '+row['kernel']+'; refresh the inventory explicitly')
    selected = [r['kernel'] for r in records]
    shapes = ['--shape-specs',args.shape_specs.resolve()] if args.shape_specs else []
    options = ['--case-options',args.case_options.resolve()] if args.case_options else []
    env = environment(POLYGEIST_NUMERIC_MODEL=args.numeric_model)
    if args.stage in ('match','all'):
        command([sys.executable, ROOT/'scripts/correctness/aten_paper115_campaign.py',
            '--output', out/'matching', '--checklist', checklist, '--cases', *selected,
            '--ranks',*[str(rank) for rank in args.ranks],
            '--split-pointwise-inputs','--jobs',str(args.jobs),
            '--frontend-support', COMPILER/'generic_solver/lib_v3/support/aten_scalar_frontend.json',
            '--region',
            '--numeric-model',args.numeric_model,'--include-validated',*shapes,*options],
            out/'match.log', env=env, timeout=21600)
    rows = report(out, selected, args.shape_specs, args.extra_gpu_results, args.scope)
    eligible = [r['kernel'] for r in rows if r['complete_match'] and r['native_baseline_available']]
    if eligible:
        runtime = [sys.executable, ROOT/'scripts/correctness/aten_fixture_batch.py',
            '--campaign', out/'matching', '--output', out/'gpu', '--cases', *eligible,
            '--rank','-1','--timing-samples','5','--timing-correctness','strict',
            '--require-full-correctness',*shapes,
            '--cudnn-plan-cache','--gpu-workspace-cache','--persistent-workspace']
        if args.stage in ('build','all'):
            command([*runtime,'--stage','build','--jobs',str(args.jobs)],
                    out/'build.log',env=env,timeout=21600,check=False)
            rows = report(out, selected, args.shape_specs, args.extra_gpu_results, args.scope)
        if args.stage in ('run','all'):
            runnable = [r['kernel'] for r in rows if r['kernel'] in eligible and
                        r['build'] and r['build'].get('rc') == 0]
            if runnable:
                first = runtime.index('--cases') + 1
                last = runtime.index('--rank')
                runtime[first:last] = runnable
                command([*runtime,'--stage','run',
                    '--library-path',setting('library_path',
                        '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu')],
                    out/'run.log',env=env,timeout=21600)
    if args.stage != 'report':
        report(out, selected, args.shape_specs, args.extra_gpu_results, args.scope)


if __name__ == '__main__':
    main()
