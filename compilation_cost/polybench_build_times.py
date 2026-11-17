#!/usr/bin/env python3
"""One sequential build of the Figure 3/4 PolyBench sets, with command timings.

Run the existing CPU/GPU benchmark builders in a pinned GPU image, overlaying
only the matcher-preparation changes. Application/runtime objects are built
afresh; prepared reference systems use an initially empty shared disk cache.
No benchmark executables are run. Native baseline builds are recorded separately.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

CHANGES = ['scripts/correctness/'+name for name in (
    'lib_v2/match_v2.py', 'region_match/__init__.py',
    'region_match/matcher.py', 'region_match/affine.py')]
IMAGE = 'ghcr.io/prontolab/looplibrarian-benchmarks:gpu-5903d7b1adb3'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inside_suite(args):
    root, output = Path('/opt/looplibrarian'), Path('/out')
    dest = output/args.suite
    dest.mkdir(exist_ok=True)
    os.environ.update(LL_ARTIFACT_OUTPUT=str(dest), LL_GPU_TARGET='x86_64',
        LL_GPU_ARCH=args.gpu_arch, CCACHE_DISABLE='1',
        REGION_MATCH_CACHE_DIR='/out/spec-cache', REGION_MATCH_NO_CACHE='0')
    sys.path[:0] = [str(root/'scripts/artifact'), str(root/'scripts/correctness')]
    import common
    os.environ.update(common.environment())
    import gpu_target
    commands, seen_cpu = [], set()
    context = {'kind': None}

    def command(argv, log, env=None, timeout=900):
        argv = list(map(str, argv))
        name = log.parent.name
        base = Path(argv[0]).name
        if context['kind']:
            kind = context['kind']
        elif base == 'polygeist_build.sh':
            kind = 'raised_compile'
            seen_cpu.add(name)
        elif base == 'build_jetson.sh':
            kind = 'raised_gpu_lower_and_link'
        elif args.suite == 'cpu' and name in seen_cpu:
            kind = 'raised_finalize'
        else:
            kind = 'native' if args.suite == 'cpu' else 'reference_cpu'
        env = dict(os.environ if env is None else env)
        if kind == 'raised_compile':
            env['POLYGEIST_MATCHER_TELEMETRY_JSONL'] = str(log.parent/'matcher.jsonl')
        number = len(commands)+1
        record_dir = dest/'commands'
        record_dir.mkdir(exist_ok=True)
        timing = record_dir/f'{number:03}.time'
        measured = ['/usr/bin/timeout', '--kill-after=10', str(timeout),
                    '/usr/bin/time', '-f', 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\npeak_rss_kib=%M',
                    '-o', str(timing), *argv]
        before = subprocess.check_output(['uptime'], text=True).strip()
        started, stages = time.monotonic(), []
        with log.open('a') as stream:
            stream.write('COMMAND '+shlex.join(argv)+'\n')
            stream.flush()
            process = subprocess.Popen(measured, cwd=root, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if re.search(r'\[\d+(?:-\d+)?/9\]', line):
                    stages.append(dict(seconds=time.monotonic()-started, label=line.strip()))
            process.stdout.close()
            rc = process.wait()
        elapsed = time.monotonic()-started
        for i, stage in enumerate(stages):
            stage['duration_seconds'] = (stages[i+1]['seconds'] if i+1<len(stages) else elapsed)-stage['seconds']
        row = dict(case=name, kind=kind, command=argv, seconds=elapsed, returncode=rc,
                   stages=stages, uptime_before=before,
                   uptime_after=subprocess.check_output(['uptime'], text=True).strip())
        commands.append(row)
        save(record_dir/f'{number:03}.json', row)
        save(dest/'command-times.json', commands)
        save(output/'progress.json', dict(suite=args.suite, case=name, command=kind,
                                         returncode=rc, seconds=elapsed, status='running'))
        return rc

    errors = {}
    if args.suite == 'cpu':
        driver = load('cpu_build', root/'polybench/cpu/reproduce.py')
        driver.command = command
        names = driver.DEFAULTS.split(',')
        options = SimpleNamespace(output=dest, target='host', openblas_root=None,
            size='LARGE', dtype='double', kernels=driver.DEFAULTS,
            preserve_views='', memory_form='cholesky')
        manifest = driver.build(options)
        records = {r['kernel']:r for r in manifest['kernels']}
    else:
        sys.path.insert(0, str(root/'polybench/gpu'))
        driver = load('build', root/'polybench/gpu/build.py')
        driver.command = command
        native = load('build_native', root/'polybench/gpu/build_native.py')
        names, records = driver.CASES, {}
        gpu_target.validate_toolkit()
        revision = subprocess.check_output(['git','-C',str(native.UP),'rev-parse','HEAD'],text=True).strip()
        if revision != '5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c':
            raise RuntimeError('Unexpected native CUDA source revision')
        for name in ('iteration_driver.c', 'paper_targets.csv'):
            shutil.copyfile(root/'polybench/gpu'/name, dest/name)
        setup_start = time.monotonic()
        subprocess.run([gpu_target.cc(), '-O2', '-c', dest/'iteration_driver.c',
                        '-o', dest/'iteration_driver.o'], check=True)
        save(dest/'setup.json', dict(iteration_driver_seconds=time.monotonic()-setup_start))
        for name in names:
            try:
                context['kind'] = None
                records[name] = driver.build(name)
                context['kind'] = 'native_cuda'
                native.build(name)
                if name in ('correlation', 'covariance'):
                    context['kind'] = 'paper_baseline_cuda'
                    sys.argv = ['build_paper_baselines.py', name]
                    runpy.run_path(str(root/'polybench/gpu/build_paper_baselines.py'))
            except Exception as error:
                errors[name] = type(error).__name__+': '+str(error)
                save(dest/name/'exception.json', dict(error=errors[name]))
            finally:
                context['kind'] = None
    rows = []
    for name in names:
        measured = [c for c in commands if c['case']==name]
        raised = [c for c in measured if c['kind'].startswith('raised')]
        proof_file = dest/name/('work' if args.suite=='cpu' else 'export')/'match.json'
        proof = json.loads(proof_file.read_text()) if proof_file.exists() else {}
        events_file = dest/name/'matcher.jsonl'
        events = [json.loads(x) for x in events_file.read_text().splitlines()] if events_file.exists() else []
        record = records.get(name, {})
        ok = (bool(raised) and bool(proof.get('emitted')) and name not in errors
              and all(c['returncode']==0 for c in raised))
        if args.suite=='cpu':ok = ok and record.get('raised_build_rc')==0
        else:ok = ok and record.get('source_build_rc')==0 and record.get('gpu_build_rc')==0
        rows.append(dict(suite=args.suite, kernel=name, status='ok' if ok else 'failed',
            raised_build_seconds=sum(c['seconds'] for c in raised),
            native_build_seconds=sum(c['seconds'] for c in measured if not c['kind'].startswith('raised')),
            all_commands_seconds=sum(c['seconds'] for c in measured),
            matching_seconds=proof.get('elapsed_ms',0)/1000 if proof else None,
            matcher_stage_seconds=sum(s['duration_seconds'] for c in raised for s in c['stages'] if s['label'].startswith('[3/9]')),
            catalog_seconds=sum(r.get('region_catalog_ms',0) for r in proof.get('diagnostics',{}).get('search',[]))/1000,
            catalog_events=[e['status'] for e in events if e['phase']=='region_catalog_cache'],
            generic_search_skipped=any(r.get('region_match_skipped')=='no remaining computation' for r in proof.get('diagnostics',{}).get('search',[])),
            calls=proof.get('emitted',[]),
            native_status='ok' if all(c['returncode']==0 for c in measured if not c['kind'].startswith('raised')) else 'failed',
            errors=errors.get(name)))
    save(dest/'build-times.json', rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default=IMAGE)
    p.add_argument('--compiler-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--gpu-arch', default='sm_86')
    p.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--suite', choices=['cpu','gpu'], help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.inside:
        inside_suite(args)
        return
    if not args.compiler_root or not args.output:p.error('--compiler-root and --output required')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    for relative in CHANGES:
        dest = out/'changes'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.compiler_root/relative, dest)
    shutil.copyfile(__file__, out/'probe.py')
    image_id = subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    request = dict(image=args.image, image_id=image_id, gpu_arch=args.gpu_arch, cpu_target='host',
                   changes={name:sha(out/'changes'/name) for name in CHANGES},
                   script_sha256=sha(out/'probe.py'),
                   cache='Initially empty prepared-reference cache shared across cases; fresh object builds.',
                   runtime_reuse_patch=False, repetitions=1, dataset='LARGE', datatype='FP64',
                   scope='Compilation only. Existing Figure 3/4 drivers; native baselines measured separately.',
                   uptime=subprocess.check_output(['uptime'],text=True).strip())
    save(out/'request.json', request)
    bootstrap = '''import json,os,subprocess,sys,shutil
from pathlib import Path
out=Path('/out');compiler=Path('/opt/looplibrarian/third_party/Polygeist')
request=json.loads((out/'request.json').read_text())
for name in request['changes']:shutil.copyfile(out/'changes'/name,compiler/name)
for suite in ['cpu','gpu']:
 subprocess.run([sys.executable,'-u','/out/probe.py','--inside','--suite',suite,'--gpu-arch',request['gpu_arch']],check=True)
'''
    (out/'bootstrap.py').write_text(bootstrap)
    start = time.monotonic()
    with (out/'driver.log').open('w') as log:
        rc = subprocess.run(['docker','run','--rm','--mount',f'type=bind,src={out},dst=/out',
                             args.image,'python3','-u','/out/bootstrap.py'],stdout=log,stderr=subprocess.STDOUT).returncode
    rows = []
    for suite in ('cpu','gpu'):
        path = out/suite/'build-times.json'
        if path.exists():rows += json.loads(path.read_text())
    save(out/'report.json', dict(request=request, results=rows, returncode=rc,
                                elapsed_seconds=time.monotonic()-start))
    if rows:
        with (out/'build-times.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    save(out/'progress.json', dict(status='complete' if rc==0 else 'failed', returncode=rc,
         cases=len(rows), successful=sum(r['status']=='ok' for r in rows)))
    print('Build time results:', out/'build-times.csv', flush=True)
    raise SystemExit(rc)


if __name__=='__main__':main()
