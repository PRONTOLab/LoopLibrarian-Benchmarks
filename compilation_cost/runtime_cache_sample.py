#!/usr/bin/env python3
"""Compare ordinary, empty-cache and reused-cache CPU GEMM builds in Docker.

Uses the Figure 3 LARGE FP64 source, whole-program compiler and output check.
Application matching, compilation and linking run anew for every build.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

BASE = ['lib_v2/match_v2.py', 'region_match/__init__.py',
        'region_match/matcher.py', 'region_match/affine.py']
CHANGES = ['lib_v2/abi_v2.py', 'lib_v2/abi_cache.py',
           'lib_v2/gen_v2_runtime.py', 'polygeist_build.sh']
TESTS = ['test_abi_persistent_cache.py', 'test_spec_abi_table_reuse.py',
         'test_fp32_blas_abi.py', 'test_region_preparation.py']
IMAGE = 'ghcr.io/prontolab/looplibrarian-benchmarks:gpu-5903d7b1adb3'


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside():
    out, root = Path('/out'), Path('/opt/looplibrarian')
    compiler = root / 'third_party/Polygeist'
    request = json.loads((out / 'request.json').read_text())
    sys.path[:0] = [str(root / 'scripts/artifact'), str(compiler / 'scripts/correctness')]
    import common
    os.environ.update(LL_GPU_TARGET='x86_64', LL_GPU_ARCH='sm_86')
    os.environ.update(common.environment())
    os.environ.update(REGION_MATCH_CACHE_DIR='/out/spec-cache',
                      POLYGEIST_ABI_CACHE_DIR='/out/abi-cache',
                      POLYGEIST_RUNTIME_CACHE_DIR='/out/object-cache',
                      REGION_MATCH_NO_CACHE='0')
    os.environ.pop('CCACHE_DISABLE', None)
    spec = importlib.util.spec_from_file_location('cpu_sample', root / 'polybench/cpu/reproduce.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    rows = []
    for relative in BASE:
        shutil.copyfile(out / 'changes' / relative, compiler / 'scripts/correctness' / relative)
    for mode in ('control', 'cold', 'warm'):
        if mode == 'cold':
            for relative in CHANGES:
                shutil.copyfile(out / 'changes' / relative, compiler / 'scripts/correctness' / relative)
        dest = out / mode
        dest.mkdir()
        os.environ['POLYGEIST_ABI_CACHE_LOG'] = str(dest / 'abi-cache.jsonl')
        commands, seen_raised = [], False

        def command(argv, log, env=None, timeout=900):
            nonlocal seen_raised
            argv = list(map(str, argv))
            raised = Path(argv[0]).name == 'polygeist_build.sh'
            kind = 'raised_compile' if raised else ('raised_finalize' if seen_raised else 'native')
            seen_raised |= raised
            before = subprocess.check_output(['uptime'], text=True).strip()
            started, stages = time.monotonic(), []
            with log.open('a') as stream:
                process = subprocess.Popen(['/usr/bin/timeout', '--kill-after=10', str(timeout), *argv],
                    cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    if re.search(r'\[\d+(?:-\d+)?/9\]', line):
                        stages.append(dict(label=line.strip(), seconds=time.monotonic()-started))
                process.stdout.close()
                rc = process.wait()
            elapsed = time.monotonic()-started
            for i, stage in enumerate(stages):
                stage['duration_seconds'] = (stages[i+1]['seconds'] if i+1<len(stages) else elapsed)-stage['seconds']
            commands.append(dict(kind=kind, command=argv, seconds=elapsed, returncode=rc,
                                 uptime_before=before, stages=stages))
            save(dest / 'commands.json', commands)
            return rc

        driver.command = command
        options = SimpleNamespace(output=dest, target='host', openblas_root=None,
            size='LARGE', dtype='double', kernels='gemm', preserve_views='', memory_form='')
        manifest = driver.build(options)
        built = manifest['kernels'][0]
        if built['native_build_rc'] or built['raised_build_rc']:
            raise RuntimeError(f'{mode} GEMM build failed; see {dest}/gemm/build.log')
        work = dest / 'gemm/work'
        proof = json.loads((work / 'match.json').read_text())
        disassembly = subprocess.check_output(['objdump', '-dr', work / 'kernel.o'], text=True)
        disassembly = '\n'.join(line for line in disassembly.splitlines() if 'file format' not in line)
        (dest / 'application.asm').write_text(disassembly)
        runtime = (work / 'spec_runtime/polygeist_v2_rt_cpu.c').read_text()
        symbol = 'polygeist_v2_cublasDgemm'
        begin = runtime.index('{', runtime.index(symbol + '('))
        depth, end = 1, begin+1
        while depth:
            depth += (runtime[end] == '{') - (runtime[end] == '}')
            end += 1
        selected = runtime[begin:end]
        (dest / 'selected-wrapper.c').write_text(selected)
        rows.append(dict(mode=mode,
            build_seconds=sum(c['seconds'] for c in commands if c['kind'].startswith('raised')),
            matching_seconds=proof['elapsed_ms']/1000, emitted=proof['emitted'],
            specification_preparation_seconds=sum(item.get('region_catalog_ms', 0)
                for item in proof.get('diagnostics', {}).get('search', []))/1000,
            abi_sha256=proof['abi_sha256'], application_sha256=sha(dest / 'application.asm'),
            selected_wrapper_sha256=sha(dest / 'selected-wrapper.c'), runtime_source_bytes=len(runtime),
            source_sha256=built['source_sha256'], stages=[s for c in commands for s in c['stages']]))
        save(out / 'report.json', dict(request=request, builds=rows))
        print(mode, round(rows[-1]['build_seconds'], 3), 'seconds', flush=True)

    checks = {field: len({r[field] if not isinstance(r[field], list) else tuple(r[field]) for r in rows}) == 1
              for field in ('emitted', 'abi_sha256', 'application_sha256',
                            'selected_wrapper_sha256', 'source_sha256')}
    if not all(checks.values()):
        raise RuntimeError(f'Build preservation check failed: {checks}')
    # Run the unchanged binaries with their existing five warmups/five samples,
    # and compare the complete final arrays. No performance claim is made here.
    dest = out / 'warm/gemm'
    for variant in ('native', 'raised'):
        with (dest / (variant + '.log')).open('w') as stream:
            subprocess.run([dest / variant], check=True, timeout=300,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', POLYGEIST_DUMP_LAST='1'),
                stdout=stream, stderr=subprocess.STDOUT)
        if '==END   DUMP_ARRAYS==' not in (dest / (variant + '.log')).read_text():
            raise RuntimeError('Missing complete output dump')
    with (dest / 'compare.log').open('w') as stream:
        subprocess.run([sys.executable, root / 'scripts/correctness/compare_polybench_dumps.py',
            dest / 'native.log', dest / 'raised.log', '--rtol', '1e-8', '--atol', '1e-8'],
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    checks['cpu_output'] = True
    for test in TESTS:
        shutil.copyfile(out / 'tests' / test, compiler / 'scripts/correctness/tests' / test)
    with (out / 'tests.log').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'unittest', *[str(Path('tests') / t) for t in TESTS]],
            cwd=compiler / 'scripts/correctness', stdout=stream, stderr=subprocess.STDOUT, check=True)
    checks['focused_tests'] = True
    with (out / 'ccache-stats.txt').open('w') as stream:
        subprocess.run(['ccache', '--show-stats'], env=dict(os.environ, CCACHE_DIR='/out/object-cache'),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    save(out / 'report.json', dict(request=request, builds=rows, checks=checks))
    print('Preservation, full-output correctness and focused tests passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--prepared-spec-cache', type=Path,
                        help='reuse an already measured mathematical specification cache in all three builds')
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    if args.inside:
        return inside()
    if not args.output or not args.compiler_root:
        parser.error('--output and --compiler-root are required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.prepared_spec_cache:
        shutil.copytree(args.prepared_spec_cache, out / 'spec-cache')
    hashes = {}
    for relative in BASE + CHANGES:
        dest = out / 'changes' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.compiler_root / 'scripts/correctness' / relative, dest)
        hashes[relative] = sha(dest)
    (out / 'tests').mkdir()
    for test in TESTS:
        shutil.copyfile(args.compiler_root / 'scripts/correctness/tests' / test, out / 'tests' / test)
    shutil.copyfile(__file__, out / 'sample.py')
    image_id = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    save(out / 'request.json', dict(image=args.image, image_id=image_id, changes=hashes,
        script_sha256=sha(out / 'sample.py'), benchmark='gemm', target='host', size='LARGE', dtype='double',
        prepared_spec_cache={str(path.relative_to(out / 'spec-cache')): sha(path)
            for path in (out / 'spec-cache').rglob('*') if path.is_file()},
        cache='ABI/runtime caches initially empty; cold prepares them, warm reuses them. Application builds remain fresh.'))
    with (out / 'driver.log').open('w') as stream:
        subprocess.run(['docker', 'run', '--rm', '--mount', f'type=bind,src={out},dst=/out',
                        args.image, 'python3', '-u', '/out/sample.py', '--inside'],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(out / 'report.json')


if __name__ == '__main__':
    main()
