#!/usr/bin/env python3
"""Build and validate complete proofs from a fresh paper-cohort campaign.

Five warmups and five alternating synchronized timing pairs are used. Failed
comparisons, incomplete matches, and differing publication statistics remain
explicit in the report. No archived binary or matching result is substituted.
"""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import aten_fixture_batch as F

ROOT = Path(__file__).resolve().parents[2]


def complete_variants(result):
    return [v for v in result.get('variants', []) if v.get('emitted') and
            v.get('abi_ok') and v.get('counts', {}).get('linalg') == 0 and
            v.get('counts', {}).get('loops') == 0]


def inventory(campaign, followups=(), names=None):
    rows = list(csv.DictReader((ROOT/'aten/paper_cases.csv').open()))
    if names is not None:
        unknown=set(names)-{r['kernel'] for r in rows}
        if unknown: raise ValueError('Unknown paper cases: '+str(sorted(unknown)))
        rows=[r for r in rows if r['kernel'] in names]
    result = []
    for row in rows:
        name = row['kernel']
        path = campaign/name/'result.json'
        if not path.is_file():
            raise ValueError('Matching campaign is not finished: '+name)
        matching = json.loads(path.read_text())
        variants = complete_variants(matching)
        selected_campaign = campaign
        for followup in followups:
            followup_path = followup/name/'result.json'
            if followup_path.is_file():
                candidate = complete_variants(json.loads(followup_path.read_text()))
                if candidate:
                    variants = candidate
                    selected_campaign = followup
        result.append(dict(kernel=name, complete_match=bool(variants),
                           native_adapter=name.removeprefix('aten_') in F.NATIVE,
                           campaign=str(selected_campaign),
                           rank=min(variants, key=lambda v: len(v['emitted']))['rank'] if variants else None))
    return result


def report(args, cases):
    paper = {r['kernel']: r for r in csv.DictReader((ROOT/'aten/paper_timings.csv').open())}
    rows = []
    for case in cases:
        row = dict(case)
        if case['complete_match']:
            proof_path = Path(case['campaign'])/case['kernel']/('rank'+str(case['rank']))/'proof.json'
            proof=json.loads(proof_path.read_text())
            row['proof_configuration'] = proof['configuration']
            row['emitted_specifications'] = proof.get('emitted',[])
        directory = args.output/case['kernel']
        for filename, key in [('build.json','build'), ('gpu-result.json','runtime')]:
            path = directory/filename
            if path.is_file():
                row[key] = json.loads(path.read_text())
        row['runtime_evidence_directory']=str(directory)
        for followup in getattr(args,'runtime_followup',[]):
            candidate=followup/case['kernel']
            if (candidate/'gpu-result.json').is_file():
                row.setdefault('previous_runtime_attempts',[]).append({
                    key:row.get(key) for key in ['build','runtime','runtime_evidence_directory']})
                row['build']=json.loads((candidate/'build.json').read_text())
                row['runtime']=json.loads((candidate/'gpu-result.json').read_text())
                row['runtime_evidence_directory']=str(candidate)
        work=Path(row['runtime_evidence_directory'])/'work'
        actual_proof=work/'match.json'
        if actual_proof.is_file():
            proof=json.loads(actual_proof.read_text())
            row['proof_configuration']=proof['configuration']
            row['emitted_specifications']=proof.get('emitted',[])
            row['runtime_proof_path']=str(actual_proof)
        abi=work/'abi.json'
        if abi.is_file():
            row['lowered_calls']=[{k:entry.get(k) for k in ['spec','call','gpu_entry']}
                for entry in json.loads(abi.read_text()).get('launches',[])]
        runtime = row.get('runtime', {})
        timing = runtime.get('timing', {})
        row['published'] = paper[case['kernel']]
        if timing.get('passed'):
            measured = timing['minimum_ms']
            historical = paper[case['kernel']]
            row['comparison'] = dict(
                raised_us=measured['raised']*1000,
                native_us=measured['native_aten_cuda']*1000,
                native_over_raised=measured['native_aten_cuda']/measured['raised'],
                raised_time_over_paper=measured['raised']*1000/float(historical['raised_us']),
                historical_native_over_raised=float(historical['native_us'])/float(historical['raised_us']),
                speedup_over_paper=(measured['native_aten_cuda']/measured['raised']) /
                    (float(historical['native_us'])/float(historical['raised_us'])),
                same_statistic=historical['timing_protocol']=='minimum of 5 after 5 warmups',
                input_kind=timing['input_kind'],
                baseline=runtime.get('scope'),
                note='Compare input recipes and native API scope before interpreting speedup differences.')
        rows.append(row)
    summary = dict(cases=len(rows), complete_matches=sum(r['complete_match'] for r in rows),
                   successful_builds=sum(r.get('build',{}).get('rc')==0 for r in rows),
                   all_requested_comparisons_passed=sum(r.get('runtime',{}).get('fully_validated',False) for r in rows),
                   timing_pairs_validated=sum(r.get('runtime',{}).get('timing',{}).get('passed',False) for r in rows),
                   original_paper_raised_passed=sum(r.get('runtime',{}).get('correctness',{}).get('paper',{}).get('raised_passed',False) for r in rows),
                   original_paper_native_passed=sum(r.get('runtime',{}).get('correctness',{}).get('paper',{}).get('native_passed') is True for r in rows),
                   original_paper_raised_native_agreement=sum(r.get('runtime',{}).get('correctness',{}).get('paper',{}).get('raised_matches_native') is True for r in rows),
                   original_paper_comparisons_passed=sum(r.get('runtime',{}).get('correctness',{}).get('paper',{}).get('passed',False) for r in rows))
    timed=[r['comparison'] for r in rows if 'comparison' in r]
    summary['raised_runtime_within_5pct_or_faster']=sum(c['raised_time_over_paper']<=1.05 for c in timed)
    summary['native_relative_speedup_within_5pct_or_better']=sum(c['speedup_over_paper']>=0.95 for c in timed)
    summary['timed_cases_with_same_publication_statistic']=sum(c['same_statistic'] for c in timed)
    F.save(args.output/'report.json', dict(summary=summary, cases=rows))
    columns=['kernel','complete_match','build_returncode','paper_raised_passed',
             'paper_native_passed','paper_raised_native_agreement','all_comparisons_passed',
             'timing_validated','timing_pairs','timing_input','timing_statistic',
             'raised_us','native_us','native_over_raised','paper_raised_us','paper_native_us',
             'raised_time_over_paper','speedup_over_paper','same_publication_statistic',
             'specifications','vendor_calls','evidence_directory']
    with (args.output/'report.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:
            runtime=row.get('runtime',{});timing=runtime.get('timing',{})
            correctness=runtime.get('correctness',{}).get('paper',{})
            comparison=row.get('comparison',{})
            samples=timing.get('samples',[])
            pairs=({s['sample'] for s in samples if s['path']=='raised'} &
                   {s['sample'] for s in samples if s['path']=='native_aten_cuda'})
            writer.writerow(dict(kernel=row['kernel'],complete_match=row['complete_match'],
                build_returncode=row.get('build',{}).get('rc'),
                paper_raised_passed=correctness.get('raised_passed'),
                paper_native_passed=correctness.get('native_passed'),
                paper_raised_native_agreement=correctness.get('raised_matches_native'),
                all_comparisons_passed=runtime.get('fully_validated'),
                timing_validated=timing.get('passed'),timing_pairs=len(pairs) if samples else None,
                timing_input=timing.get('input_kind'),timing_statistic=timing.get('statistic'),
                raised_us=comparison.get('raised_us'),native_us=comparison.get('native_us'),
                native_over_raised=comparison.get('native_over_raised'),
                paper_raised_us=row['published'].get('raised_us'),
                paper_native_us=row['published'].get('native_us'),
                raised_time_over_paper=comparison.get('raised_time_over_paper'),
                speedup_over_paper=comparison.get('speedup_over_paper'),
                same_publication_statistic=comparison.get('same_statistic'),
                specifications=';'.join(row.get('emitted_specifications',[])),
                vendor_calls=';'.join(str(call.get('gpu_entry') or call.get('call'))
                                      for call in row.get('lowered_calls',[])),
                evidence_directory=row['runtime_evidence_directory']))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--followup', type=Path, action='append', default=[],
                        help='Additional fresh matching campaign; retain initial evidence and prefer its complete proofs')
    parser.add_argument('--runtime-followup',type=Path,action='append',default=[],
                        help='Later build/run evidence for the report; retain earlier failures alongside it')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases',nargs='+',help='Explicit paper-cohort subset; default is all 115 cases')
    parser.add_argument('--stage', choices=['build','run','report'], required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--ready-only', action='store_true',help='Run only cases whose executable build has finished')
    parser.add_argument('--resume', action='store_true',help='Keep existing runtime evidence for the same binary and options')
    parser.add_argument('--transport')
    parser.add_argument('--remote-library-path', default='/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu')
    parser.add_argument('--cutensor-root', type=Path)
    parser.add_argument('--native-paper-inputs', action='store_true',
                        help='Where separately recorded, also test and time native-harness inputs; keep original raised-input failures')
    parser.add_argument('--cudnn-plan-cache', action='store_true')
    parser.add_argument('--gpu-workspace-cache', action='store_true')
    parser.add_argument('--persistent-workspace', action='store_true')
    args = parser.parse_args()
    args.campaign = args.campaign.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = inventory(args.campaign, [p.resolve() for p in args.followup], args.cases)
    selected = [r for r in cases if r['complete_match'] and r['native_adapter']]
    F.save(args.output/'selection.json', cases)
    if args.stage == 'build':
        def build(case):
            config = SimpleNamespace(**vars(args), rank=case['rank'],
                                     shape_specs=ROOT/'aten/paper_shapes.json', timing_samples=5)
            config.campaign = Path(case['campaign'])
            try:
                return F.build(case['kernel'], config)
            except Exception as error:
                result = dict(kernel=case['kernel'],rc=1,error=type(error).__name__+': '+str(error))
                F.save(args.output/case['kernel']/'build.json', result)
                print(case['kernel'], result['error'], flush=True)
                return result
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            results = list(pool.map(build, selected))
        F.save(args.output/'build-results.json', results)
    elif args.stage == 'run':
        groups = {'paper': [], 'native-paper': []}
        for case in selected:
            build_path=args.output/case['kernel']/'build.json'
            if args.ready_only and not build_path.is_file():
                continue
            built = json.loads(build_path.read_text())
            alternative = args.native_paper_inputs and 'native_paper_initializers' in built.get('configuration',{})
            previous=args.output/case['kernel']/'gpu-result.json'
            if args.resume and previous.is_file():
                saved=json.loads(previous.read_text())
                if saved.get('binary_sha256')==built.get('binary_sha256') and (
                        built.get('rc') or
                        (saved.get('cudnn_plan_cache')==args.cudnn_plan_cache and
                         saved.get('gpu_workspace_cache')==args.gpu_workspace_cache and
                         set(saved.get('correctness',{}))==({'paper','native-paper'} if alternative else {'paper'}))):
                    continue
                raise ValueError('Existing runtime evidence differs; use a fresh output: '+case['kernel'])
            groups['native-paper' if alternative else 'paper'].append(case['kernel'])
        for recipe, names in groups.items():
            if not names:
                continue
            config = SimpleNamespace(**vars(args), run_timeout=600, skip_timing=False,
                correctness_kinds=['paper']+(['native-paper'] if recipe=='native-paper' else []),
                timing_input=recipe, time_validated_normal_inputs=False)
            F.run_gpu(names, config)
    report(args, cases)


if __name__ == '__main__':
    main()
