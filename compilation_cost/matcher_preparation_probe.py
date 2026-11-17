#!/usr/bin/env python3
"""Measure empty-search avoidance and prepared-spec reuse in the compiler image.

Uses the runtime reuse probe's checked runtime/ABI setup, then measures fresh
application builds before the matcher changes and twice after them. A separate
measurement loads the CUDA reference set in fresh processes with cold and warm
caches. This is a two-input diagnostic, not the complete compilation-cost table.
No GPU execution is performed.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import runtime_reuse_probe as R

CHANGES = ['scripts/correctness/'+name for name in (
    'lib_v2/match_v2.py', 'region_match/__init__.py',
    'region_match/matcher.py', 'region_match/affine.py')]


def measure_catalog(compiler, out):
    """Time the same CUDA reference set in two independent Python processes."""
    allowed = {entry['name'] for path in (compiler/'generic_solver/lib_v3/manifest').glob('*.json')
               for entry in json.loads(path.read_text()).get('specs', [])}
    proof = json.loads((out/'before/polybench/gemm/full-1/work/match.json').read_text())
    allowed -= set(proof['rejected_specifications'])
    R.save(out/'catalog-allowed.json', sorted(allowed))
    child = '''import json,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/scripts/correctness')
from region_match.matcher import load_specs
allowed=json.loads(Path(sys.argv[2]).read_text())
start=time.monotonic()
specs=load_specs(sys.argv[1]+'/generic_solver/lib_v3',allowed_specs=allowed)
seconds=time.monotonic()-start
rows=[dict(name=s.name,stage=s.source_stage,validation=s.validation,
           fixed_ints=s.fixed_ints,argnames=s.argnames,argtypes=s.argtypes,dims=s.dims,
           equations=repr(s.system.equations)) for s in specs]
Path(sys.argv[3]).write_text(json.dumps(dict(seconds=seconds,specifications=rows),indent=2))
'''
    (out/'catalog-child.py').write_text(child)
    records = []
    for phase, seed in [('cold', '19'), ('warm', '71')]:
        dest = out/('catalog-'+phase+'.json')
        env = dict(os.environ, REGION_MATCH_CACHE_DIR=str(out/'catalog-probe-cache'),
                   REGION_MATCH_NO_CACHE='0', POLYGEIST_NUMERIC_MODEL='relaxed', EQSAT_ABLATION='full',
                   POLYGEIST_SATURATION_TIMEOUT='10', POLYGEIST_MATCHER_COLLECT_SIZES='1',
                   POLYGEIST_MATCHER_TELEMETRY_JSONL=str(out/('catalog-'+phase+'.jsonl')),
                   PYTHONHASHSEED=seed)
        R.run([sys.executable, out/'catalog-child.py', compiler, out/'catalog-allowed.json', dest],
              env=env, timeout=360)
        records.append(json.loads(dest.read_text()))
    if records[0]['specifications'] != records[1]['specifications']:
        raise RuntimeError('Prepared reference systems changed across processes')
    result = dict(cold_seconds=records[0]['seconds'], warm_seconds=records[1]['seconds'],
                  eligible_specifications=len(allowed), loaded_specifications=len(records[0]['specifications']),
                  reference_systems_unchanged=True)
    R.save(out/'catalog-summary.json', result)
    return result


def outcome(work):
    proof = json.loads((work/'match.json').read_text())
    return dict(application=R.app_code(work), emitted=proof['emitted'],
                abi=proof['abi_sha256'],
                wrappers=R.sha(work/'spec_runtime/polygeist_v2_rt_cuda.c'))


def inside(args):
    root = Path('/opt/looplibrarian')
    compiler = root/'third_party/Polygeist'
    out = Path('/out')
    os.environ.update(LL_ARTIFACT_CONFIG='/out/config.json', LL_GPU_TARGET='jetson',
        CCACHE_DIR='/out/object-cache', CCACHE_COMPILERCHECK='content',
        CCACHE_BASEDIR='/out', CCACHE_NOHASHDIR='true', CCACHE_MAXSIZE='2G')
    sys.path[:0] = [str(root/'scripts/artifact'), str(root/'scripts/correctness'),
                   str(compiler/'scripts/correctness'), str(compiler/'scripts/correctness/lib_v2')]
    shutil.copyfile(out/'reproduce.py', root/'compilation_cost/reproduce.py')
    spec = importlib.util.spec_from_file_location('cost_driver', root/'compilation_cost/reproduce.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    os.environ.update(driver.environment())
    import spec_provenance
    errors, error = spec_provenance.validate(compiler/'generic_solver/lib_v3/import_manifest.json')
    stale = {k:v for k,v in errors.items() if not v.startswith('reference import has no ')}
    if error or stale:
        raise RuntimeError('Invalid imports: '+str(error or stale))
    available = json.loads((root/'compilation_cost/inputs.json').read_text())['cases']
    cases = [next(c for c in available if c['suite']+'/'+c['kernel']==name) for name in R.DEFAULT_CASES]
    build = compiler/'scripts/correctness/polygeist_build.sh'
    generator = compiler/'scripts/correctness/lib_v2/gen_v2_runtime.py'
    abi = compiler/'scripts/correctness/lib_v2/abi_v2.py'
    build.write_text(R.runtime_patch(build.read_text()))
    generator.write_text(R.generator_patch(generator.read_text()))
    abi.write_text(R.replace_once(abi.read_text(), '\ndef main() -> int:',
                                 '\n'+R.ABI_CACHE+'\ndef main() -> int:'))
    inputs = R.fingerprint(compiler)
    baseline = Path('/baseline')
    expected = json.loads((baseline/'metadata_warm/inputs.json').read_text())
    if inputs != expected:
        raise RuntimeError('Image/runtime setup differs from retained control')
    shutil.copytree(baseline/'object-cache', out/'object-cache')
    os.environ['LL_PROBE_ABI_CACHE'] = str(out/'metadata-cache')
    reports, controls = [], {}
    for phase in ('before', 'after_cold', 'after_warm'):
        started = time.monotonic()
        if phase == 'after_cold':
            for relative in CHANGES:
                shutil.copyfile(out/'changes'/relative, compiler/relative)
        inputs = R.fingerprint(compiler)
        key = R.hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        os.environ['LL_PROBE_INPUT_KEY'] = key
        R.save(out/phase/'inputs.json', inputs)
        # Hold unrelated ABI preparation outside all three application timings.
        R.run([sys.executable, '-c', 'import abi_v2; abi_v2.build_table()'],
              stdout=subprocess.DEVNULL)
        R.save(out/phase/'setup.json', dict(seconds=time.monotonic()-started))
        state = {Path(p):(Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in inputs}
        cache = out/('before-spec-cache' if phase=='before' else 'prepared-spec-cache')
        for case in cases:
            options = SimpleNamespace(output=out/phase, timeout=args.timeout,
                                      compiler_state=state, spec_cache=cache)
            driver.measure(case, 'full', 1, options)
            directory = out/phase/case['suite']/case['kernel']/'full-1'
            result = json.loads((directory/'result.json').read_text())
            if result['status'] != 'ok':
                raise RuntimeError(f'{phase}/{case["kernel"]}: {result["status"]}')
            current = outcome(directory/'work')
            name = case['suite']+'/'+case['kernel']
            if phase == 'before':
                controls[name] = current
                previous = outcome(baseline/'metadata_warm'/case['suite']/case['kernel']/'full-1/work')
                if current != previous:
                    raise RuntimeError('Fresh baseline differs: '+name)
            if current != controls[name]:
                R.save(directory/'changed-output.json', dict(before=controls[name], after=current))
                raise RuntimeError('Generated computation or wrappers changed: '+name)
            events = [json.loads(line) for line in (directory/'matcher.jsonl').read_text().splitlines()]
            proof = json.loads((directory/'work/match.json').read_text())
            row = dict(phase=phase, case=name, seconds=result['wall_seconds'],
                       matching_ms=result['matching_ms'], output_unchanged=True,
                       catalog_events=[e for e in events if e['phase']=='region_catalog_cache'],
                       region_proofs=sum(e['phase']=='region_proof' for e in events),
                       diagnostics=proof.get('diagnostics', []), uptime=result['uptime_after'])
            reports.append(row)
            R.save(out/'progress.json', dict(status='running', results=reports))
    catalog = measure_catalog(compiler, out)
    report = dict(status='complete', results=reports, catalog=catalog,
                  scope='Fresh application builds; warmed runtime objects and ABI metadata. No execution.',
                  cache='Builds share each version\'s spec-cache directory if needed; '
                        'the separate catalog measurement exercises cold and warm preparation.')
    R.save(out/'report.json', report)
    R.save(out/'progress.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default=R.IMAGE)
    p.add_argument('--compiler-root', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--sdk-root', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--timeout', type=int, default=360)
    p.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.inside:
        inside(args)
        return
    if not all((args.compiler_root, args.baseline, args.sdk_root, args.config, args.output)):
        p.error('--compiler-root, --baseline, --sdk-root, --config, --output are required')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    image = R.run(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'],
                  capture_output=True, text=True).stdout.strip()
    old = json.loads((args.baseline/'request.json').read_text())
    if image != old['image_id'] or R.sha(args.config) != old['config_sha256'] or old['cases'] != R.DEFAULT_CASES:
        p.error('Baseline image, configuration and inputs must agree')
    for relative in CHANGES:
        dest = out/'changes'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.compiler_root/relative, dest)
    for name in ('runtime_reuse_probe.py', 'reproduce.py'):
        shutil.copyfile(Path(__file__).with_name(name), out/name)
    shutil.copyfile(__file__, out/'probe.py')
    shutil.copyfile(args.config, out/'config.json')
    R.save(out/'request.json', dict(image_id=image, cases=R.DEFAULT_CASES,
        baseline=str(args.baseline.resolve()), config_sha256=R.sha(args.config),
        changes={name:R.sha(out/'changes'/name) for name in CHANGES},
        uptime=R.run(['uptime'], capture_output=True, text=True).stdout.strip()))
    command = ['docker', 'run', '--rm',
        '--mount', f'type=bind,src={args.sdk_root.resolve()},dst=/sdk,readonly',
        '--mount', f'type=bind,src={args.baseline.resolve()},dst=/baseline,readonly',
        '--mount', f'type=bind,src={out},dst=/out', args.image,
        'python3', '-u', '/out/probe.py', '--inside', '--timeout', str(args.timeout)]
    with (out/'driver.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit('Probe failed; inspect '+str(out/'driver.log'))
    report = json.loads((out/'report.json').read_text())
    summary = '\n'.join(f'{r["phase"]:12} {r["case"]:24} {r["seconds"]:.3f} s '
                        f'matching {r["matching_ms"]/1000:.3f} s; identical output'
                        for r in report['results'])+'\n'
    catalog = report['catalog']
    summary += (f'Prepared reference systems: {catalog["cold_seconds"]:.3f} s cold, '
                f'{catalog["warm_seconds"]:.3f} s in a fresh process using the cache; unchanged.\n'
                'Build totals use the runtime/ABI reuse setup from runtime_reuse_probe.py.\n'
                'Application compilation and matching are performed afresh. No GPU execution.\n')
    (out/'summary.txt').write_text(summary)
    print(summary)


if __name__ == '__main__':
    main()
