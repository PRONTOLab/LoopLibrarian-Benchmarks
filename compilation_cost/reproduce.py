"""Fresh, sequential source-to-linked-executable compilation measurements."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/artifact'),str(ROOT/'scripts/correctness')]
from common import COMPILER, OUT, environment, provenance, save, sha


def proof_measurements(path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    linalg=[r for r in rows if r['phase']=='linalg_match']
    region=[r for r in rows if r['phase']=='region_proof']
    return dict(records=len(rows),
        saturation_seconds=sum(r.get('saturation_elapsed_ms',0) for r in rows)/1000,
        proof_seconds=sum(r.get('proof_elapsed_ms',0) for r in rows)/1000,
        graph_collection_seconds=sum(r.get('telemetry_elapsed_ms',0) for r in rows)/1000,
        iterations=sum(r.get('proof_iterations',0) for r in linalg)+sum(r['iterations'] for r in region),
        rule_matches=sum(r.get('rule_matches',0) for r in rows),
        egraph_nodes_sum=sum(r.get('egraph_nodes_sum',0) for r in linalg)+sum(r['egraph_nodes'] for r in region),
        egraph_classes_sum=sum(r.get('egraph_classes_sum',0) for r in linalg)+sum(r['egraph_classes'] for r in region),
        egraph_nodes_max=max([r.get('egraph_nodes_max',0) for r in linalg]+[r['egraph_nodes'] for r in region]+[0]),
        egraph_classes_max=max([r.get('egraph_classes_max',0) for r in linalg]+[r['egraph_classes'] for r in region]+[0]),
        timeouts=sum(r.get('proof_timeouts',0) for r in linalg)+sum(r['timeout'] for r in region),
        worker_errors=sum(r.get('proof_worker_errors',0)+r.get('distributive_worker_errors',0) for r in linalg)+
                      sum(r['worker_error'] is not None for r in region))


def distributions(values):
    if not values:
        return None
    q1, _, q3 = statistics.quantiles(values,n=4,method='inclusive') if len(values)>1 else [values[0]]*3
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values),
                q1=q1,q3=q3,iqr=q3-q1,samples=len(values))


def measured_fields(row):
    fields={'wall_seconds':row['wall_seconds']}
    fields.update({key:value for key,value in row.get('process_timing',{}).items()
                   if key in ['user_seconds','system_seconds','peak_rss_kib']})
    fields.update({'proof_'+key:value for key,value in row.get('proof_measurements',{}).items()
                   if key not in ['records','timeouts','worker_errors']})
    if row.get('matching_ms') is not None:
        fields['matcher_report_seconds']=row['matching_ms']/1000
    labels={'1':'frontend_seconds','2':'raising_seconds','3':'matching_stage_seconds',
            '4-5':'abi_seconds','6':'llvm_lowering_seconds','7':'abi_bridge_seconds',
            '8':'runtime_and_object_compilation_seconds','9':'link_seconds'}
    for event in row.get('stages',[]):
        match=re.search(r'\[([\d-]+)/9\]',event['label'])
        if match and match[1] in labels:
            key=labels[match[1]]
            fields[key]=fields.get(key,0)+event['duration_seconds']
    return fields


def summarize(output, cases, repetitions):
    records=[]
    for case in cases:
        for mode in ['full','none']:
            rows=[]
            for n in range(1,repetitions+1):
                p=output/case['suite']/case['kernel']/(mode+'-'+str(n))/'result.json'
                if p.exists(): rows.append(json.loads(p.read_text()))
            good=[r for r in rows if r['status']=='ok']
            first_three=[r for r in rows if r['repetition'] in (1,2,3)]
            measurements=[measured_fields(r) for r in good]
            keys=sorted({k for measurement in measurements for k in measurement})
            metrics={key:distributions([r[key] for r in measurements if key in r]) for key in keys}
            records.append(dict(suite=case['suite'],kernel=case['kernel'],mode=mode,
                completed=len(rows),successful=len(good),requested=repetitions,
                status='complete' if len(good)==repetitions else 'incomplete',
                median_seconds=statistics.median(r['wall_seconds'] for r in good) if good else None,
                first_three_median_seconds=statistics.median(r['wall_seconds'] for r in first_three)
                    if len(first_three)==3 and all(r['status']=='ok' for r in first_three) else None,
                metrics=metrics,
                proof_timeouts=sum(r.get('proof_measurements',{}).get('timeouts',0) for r in rows),
                proof_worker_errors=sum(r.get('proof_measurements',{}).get('worker_errors',0) for r in rows),
                emitted_calls_by_repetition=[dict(repetition=r['repetition'],calls=r.get('emitted',[])) for r in rows],
                failures=[dict(repetition=r['repetition'],status=r['status'],returncode=r['returncode'])
                          for r in rows if r['status']!='ok']))
    tables=[]
    for suite in sorted({c['suite'] for c in cases}):
        for mode in ['full','none']:
            selected=[r for r in records if r['suite']==suite and r['mode']==mode]
            values=[r['median_seconds'] for r in selected if r['status']=='complete']
            tables.append(dict(suite=suite,mode=mode,requested_inputs=len(selected),
                complete_inputs=len(values),incomplete_inputs=len(selected)-len(values),
                mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None))
    first_three_tables=[]
    for table in tables:
        selected=[r for r in records if r['suite']==table['suite'] and r['mode']==table['mode']]
        values=[r['first_three_median_seconds'] for r in selected
                if r['first_three_median_seconds'] is not None]
        first_three_tables.append(dict(suite=table['suite'],mode=table['mode'],repetitions=3,
            requested_inputs=len(selected),complete_inputs=len(values),
            incomplete_inputs=len(selected)-len(values),
            mean=statistics.mean(values) if values else None,
            median=statistics.median(values) if values else None,
            minimum=min(values) if values else None,maximum=max(values) if values else None))
    metric_tables=[]
    for table in tables:
        selected=[r for r in records if r['suite']==table['suite'] and r['mode']==table['mode']]
        keys=sorted({key for row in selected for key in row['metrics']})
        for key in keys:
            values=[r['metrics'][key]['median'] for r in selected if r['status']=='complete'
                    and key in r['metrics'] and r['metrics'][key]['samples']==repetitions]
            metric_tables.append(dict(suite=table['suite'],mode=table['mode'],metric=key,
                requested_inputs=len(selected),complete_inputs=len(values),
                incomplete_inputs=len(selected)-len(values),
                mean=statistics.mean(values) if values else None,
                **(distributions(values) or dict(median=None,minimum=None,maximum=None,q1=None,q3=None,iqr=None,samples=0))))
    for filename,rows in [('compilation-table.csv',tables),
                          ('first-three-compilation-table.csv',first_three_tables),
                          ('measurement-tables.csv',metric_tables)]:
        with (output/filename).open('w',newline='') as stream:
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(output/'summary.json',dict(tables=tables,first_three_tables=first_three_tables,
        metric_tables=metric_tables,inputs=records,
        scope='Complete C-to-linked-AArch64 builds; per-input medians. Failed/incomplete inputs remain explicit.',
        publication_protocol='Paper used three repetitions; first-three medians are also retained.',
        experiment_protocol=f'{repetitions} fresh repetitions in each EqSat arm; no compiler-result cache.'))
    return tables


def measure(case, mode, repetition, args):
    if any((p.stat().st_size,p.stat().st_mtime_ns)!=state for p,state in args.compiler_state.items()):
        raise RuntimeError('Compiler inputs changed during measurement; start a fresh campaign')
    dest=args.output/case['suite']/case['kernel']/(mode+'-'+str(repetition))
    if (dest/'result.json').exists(): return
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Interrupted build directory retained; use a fresh output: '+str(dest))
    dest.mkdir(parents=True,exist_ok=True)
    source=ROOT/case['source'];function=case['function']
    original_source=source
    options=case.get('match_options',{})
    if case['suite']=='aten':
        from aten_paper_shapes import load_shapes, materialize
        source=dest/original_source.name
        materialize(original_source,source,load_shapes(ROOT/'aten/paper_shapes.json')[case['kernel']])
    harness=dest/'link_harness.c'
    harness.write_text('extern void '+function+'(void);\nstatic void (*volatile keep)(void)='+function+';\nint main(void){return keep==0;}\n')
    executable=dest/'program';timefile=dest/'time.txt';work=dest/'work'
    build=[str(COMPILER/'scripts/correctness/polygeist_build.sh'),'--target=jetson',
           '--function='+function,'--harness='+str(harness),'-o',str(executable),str(source)]
    build += [flag.replace('{root}',str(ROOT)) for flag in case['cflags']]
    if options.get('matching_stage')=='linalg':build.append('--no-debuf')
    command=['/usr/bin/timeout','--kill-after=10',str(args.timeout),'/usr/bin/time','-f',
             'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\npeak_rss_kib=%M',
             '-o',str(timefile),*build]
    env=environment(PYTHON=sys.executable,EQSAT_ABLATION=mode,
        POLYGEIST_BUILD_WORK_DIR=work,POLYGEIST_ABI_BACKEND='cuda',
        POLYGEIST_NUMERIC_MODEL=options.get('numeric_model','relaxed'),CCACHE_DISABLE=1,
        POLYGEIST_MATCHER_TELEMETRY_JSONL=dest/'matcher.jsonl',
        POLYGEIST_MATCHER_COLLECT_SIZES=1,POLYGEIST_SATURATION_TIMEOUT=10,
        REGION_MATCH_CACHE_DIR=args.output/'catalog-cache'/mode)
    rank=options.get('ranks',[0])[0]
    env.update(POLYGEIST_MATCH_MAX_OPERAND_RANK=str(rank),
        POLYGEIST_MATCH_FUSE_INITIALIZERS=str(int(rank>0)),
        POLYGEIST_MATCH_REGION=str(int(options.get('region',True))),
        POLYGEIST_MATCH_REGION_BUDGET=str(options.get('region_budget',60)),
        POLYGEIST_MATCH_SPLIT_REDUCTION_INPUTS=str(int(options.get('split_reduction_inputs',False))),
        POLYGEIST_MATCH_SPLIT_POINTWISE_INPUTS=str(int(options.get('split_pointwise_inputs',False))),
        POLYGEIST_MATCH_ONLY_KERNELS=' '.join(options.get('specifications') or []))
    env.pop('POLYGEIST_MATCH_PROOF',None)
    env.pop('POLYGEIST_SEMANTIC_MLIR',None)
    save(dest/'request.json',dict(case=case,mode=mode,repetition=repetition,command=command,
        source_sha256=sha(source),original_source_sha256=sha(original_source),
        harness_sha256=sha(harness),matching_options=options,load_before=os.getloadavg(),
        uptime_before=subprocess.check_output(['uptime'],text=True).strip()))
    started=time.monotonic();events=[]
    with (dest/'build.log').open('w') as log:
        process=subprocess.Popen(command,env=env,cwd=ROOT,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush()
            if re.search(r'\[\d+(?:-\d+)?/9\]',line):
                events.append(dict(seconds=time.monotonic()-started,label=line.strip()))
        rc=process.wait()
    elapsed=time.monotonic()-started
    timing={}
    if timefile.exists():
        for line in timefile.read_text().splitlines():
            if '=' in line:
                key,value=line.split('=',1)
                try: timing[key]=float(value)
                except ValueError: pass
    for index,event in enumerate(events):
        event['duration_seconds']=(events[index+1]['seconds'] if index+1<len(events) else elapsed)-event['seconds']
    proof=json.loads((work/'match.json').read_text()) if (work/'match.json').exists() else {}
    status='ok' if rc==0 and executable.is_file() else 'timeout' if rc in [124,137] else 'error'
    metrics=proof_measurements(dest/'matcher.jsonl')
    if status=='ok' and not metrics['records']: status='missing_metrics'
    row=dict(suite=case['suite'],kernel=case['kernel'],mode=mode,repetition=repetition,
        status=status,returncode=rc,wall_seconds=elapsed,process_timing=timing,
        stages=events,executable_sha256=sha(executable) if executable.exists() else None,
        emitted=proof.get('emitted',[]),matching_ms=proof.get('elapsed_ms'),
        telemetry_available=bool(metrics['records']),proof_measurements=metrics,
        saturation_time_complete=metrics['timeouts']==0 and metrics['worker_errors']==0,
        load_after=os.getloadavg(),
        uptime_after=subprocess.check_output(['uptime'],text=True).strip())
    save(dest/'result.json',row)
    print(case['suite'],case['kernel'],mode,repetition,status,round(elapsed,3),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['build','run','summarize','all'],default='all')
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--repetitions',type=int,help='Fresh repetitions per arm (default: 5; summarize uses recorded count)')
    p.add_argument('--timeout',type=int,default=300)
    p.add_argument('--suites',nargs='+',choices=['polybench','llama','aten','mfem'])
    p.add_argument('--cases',nargs='+',help='kernel names or suite/kernel names')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.output=args.output.resolve()
    if args.stage=='summarize':
        metadata=args.output/'measurement.json'
        if not metadata.is_file():p.error('missing measurement.json in the requested output')
        recorded=json.loads(metadata.read_text())
        if args.repetitions is not None and args.repetitions!=recorded['repetitions']:
            p.error('summary repetition count must match the recorded experiment')
        if args.suites or args.cases:
            p.error('summarize retains the recorded cohort; inspect per-input rows for subsets')
        tables=summarize(args.output,recorded['cases'],recorded['repetitions'])
        print(json.dumps(tables),flush=True)
        return
    if args.repetitions is None:args.repetitions=5
    if args.repetitions<1: p.error('positive repetitions required')
    manifest=Path(__file__).with_name('inputs.json')
    cases=[c for c in json.loads(manifest.read_text())['cases']
        if (not args.suites or c['suite'] in args.suites) and
           (not args.cases or c['kernel'] in args.cases or c['suite']+'/'+c['kernel'] in args.cases)]
    if not cases:p.error('no selected inputs')
    if args.cases:
        known={name for c in cases for name in (c['kernel'],c['suite']+'/'+c['kernel'])}
        unknown=set(args.cases)-known
        if unknown:p.error('unknown inputs: '+str(sorted(unknown)))
    missing=[c['source'] for c in cases if not (ROOT/c['source']).is_file()]
    if missing:p.error('missing inputs; run dependency setup: '+str(missing))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage!='summarize':
        # Missing public dependencies must not silently time an empty catalog.
        os.environ.update(environment())
        sys.path.insert(0,str(COMPILER/'scripts/correctness/lib_v2'))
        import spec_provenance
        errors,common_error=spec_provenance.validate(COMPILER/'generic_solver/lib_v3/import_manifest.json')
        stale={k:v for k,v in errors.items() if not v.startswith('reference import has no ')}
        if common_error or stale:
            p.error('Invalid specification imports; prepare dependencies and rebuild the catalog: '+
                    str(common_error or list(stale.items())[:3]))
        metadata=args.output/'measurement.json'
        compiler_files=[COMPILER/'build/bin/cgeist',COMPILER/'build/bin/polygeist-opt',
            COMPILER/'generic_solver/kernel_library_v3.mlir',COMPILER/'generic_solver/lib_v3/import_manifest.json']
        compiler_files += sorted(p for folder in ['scripts/correctness','runtime']
            for p in (COMPILER/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.sh','.c','.cu','.h','.inc'])
        args.compiler_state={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in compiler_files}
        config=dict(cases=cases,repetitions=args.repetitions,timeout=args.timeout,
            compiler=str(COMPILER),manifest_sha256=sha(manifest),
            compiler_input_hashes={str(p.relative_to(COMPILER)):sha(p) for p in compiler_files},
            driver_sha256=sha(Path(__file__)),source_hashes={c['source']:sha(ROOT/c['source']) for c in cases},
            paper_shapes_sha256=sha(ROOT/'aten/paper_shapes.json'),
            sequential=True,cache_disabled=True,numeric_model='ATen per-case policy; relaxed for other suites',modes=['full','none'],
            saturation_timeout_seconds=10,deadline_policy='isolated proof process; process overhead included',
            graph_counting='serialized graph nodes and classes including primitive values',
            region_catalog_cache='fresh campaign-local cache per EqSat arm; reused between builds',
            rss_scope='GNU time maximum resident set of a process, not the sum of simultaneous processes')
        if metadata.exists():
            if not args.resume or json.loads(metadata.read_text())!=config:
                p.error('use fresh output or --resume with exactly the same experiment')
        else:
            save(metadata,config);provenance(args.output)
            (args.output/'measurement_driver.py').write_bytes(Path(__file__).read_bytes())
            (args.output/'inputs.json').write_bytes(manifest.read_bytes())
            (args.output/'aten-paper-shapes.json').write_bytes((ROOT/'aten/paper_shapes.json').read_bytes())
        for repetition in range(1,args.repetitions+1):
            for index,case in enumerate(cases):
                modes=['full','none'] if (index+repetition)%2 else ['none','full']
                for mode in modes: measure(case,mode,repetition,args)
                summarize(args.output,cases,args.repetitions)
    tables=summarize(args.output,cases,args.repetitions)
    print(json.dumps(tables),flush=True)


if __name__=='__main__':main()
