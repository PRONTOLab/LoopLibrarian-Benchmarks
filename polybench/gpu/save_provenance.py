#!/usr/bin/env python3
"""Preserve the source inputs and tool hashes for a Figure 4 run."""
import json
import shutil
import subprocess
from pathlib import Path

from ll_paths import COMPILER, DEPENDENCIES
from common import OUT, ROOT, sha


def snapshot(files, root, destination, prefix=''):
    """Copy one source tree, keeping compiler-relative keys used by the audit."""
    hashes = {}
    for path in sorted(set(files)):
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[prefix + relative.as_posix()] = sha(path)
    return hashes


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                   text=True).strip()


def verify_build_inputs(output, compiler):
    """Do not replace a build's source record with files from another image."""
    actual = {}
    for case in json.loads((output / 'cases.json').read_text()):
        build = json.loads((output / case / 'build.json').read_text())
        proof_path = output / case / 'export/match.json'
        proof = json.loads(proof_path.read_text())
        if sha(proof_path) != build['proof_sha256']:
            raise RuntimeError(f'{case}: matching proof changed since build')
        expected = dict(proof['semantic_inputs'])
        expected['build/bin/polygeist-opt'] = build['compiler_sha256']
        expected[proof['library']] = proof['library_sha256']
        for name, digest in expected.items():
            if name not in actual:
                actual[name] = sha(compiler / name)
            if actual[name] != digest:
                raise RuntimeError(f'{case}: {name} differs from the build; '
                                   'use the original build image or rebuild the experiment')


def main():
    verify_build_inputs(OUT, COMPILER)
    dest = OUT / 'source-snapshot-final'
    here = Path(__file__).resolve().parent
    benchmark_files = list((ROOT / 'scripts/correctness').rglob('*.py'))
    benchmark_files += list(here.glob('*.py'))
    benchmark_files += [here / 'iteration_driver.c', here / 'paper_targets.csv']
    benchmark_files += [ROOT / 'scripts/correctness' / name for name in
                        ('polygeist_build.sh', 'build_jetson.sh')]
    compiler_files = [COMPILER / name for name in (
        'runtime/polygeist_v2_rt_cuda.c', 'runtime/polygeist_cublas_rt_cuda.c',
        'runtime/polygeist_v2_rt.h', 'runtime/polygeist_v2_rt_cpu.c',
        'runtime/polygeist_cublas_rt.h', 'generic_solver/kernel_library_v3.mlir',
        'generic_solver/lib_v3/import_manifest.json',
        'lib/polygeist/Passes/PlanGpuDataResidency.cpp',
        'lib/polygeist/Passes/PrepareGpuResidualPipeline.cpp',
        'lib/polygeist/Passes/InstrumentGpuRegionTiming.cpp',
        'include/polygeist/Passes/Passes.td')]
    compiler_files += list((COMPILER / 'generic_solver/lib_v3/manifest').glob('*.json'))
    compiler_files += list((COMPILER / 'generic_solver/lib_v3/ref_c').glob('*.c'))
    for case in json.loads((OUT / 'cases.json').read_text()):
        proof = json.loads((OUT / case / 'export/match.json').read_text())
        compiler_files += [COMPILER / p for p in proof.get('semantic_inputs', {})]
    hashes = snapshot(benchmark_files, ROOT, dest / 'benchmarks', 'benchmarks/')
    hashes.update(snapshot(compiler_files, COMPILER, dest / 'compiler'))
    for name in ('build/bin/cgeist', 'build/bin/polygeist-opt',
                 'llvm-project/build/bin/mlir-opt', 'llvm-project/build/bin/mlir-translate',
                 'llvm-project/build/bin/clang'):
        hashes[name] = sha(COMPILER / name)
    record = dict(
        head=revision(ROOT), compiler_commit=revision(COMPILER),
        host_uptime=subprocess.check_output(['uptime'], text=True).strip(), hashes=hashes,
        upstream_polybench_gpu_commit=revision(DEPENDENCIES / 'polybenchGpu'),
        protocol=dict(warmups=5, samples=5, statistic='minimum', dataset='LARGE', datatype='FP64',
                      rtol=1e-8, atol=1e-8, timing_scope='CUDA-event compute interval',
                      synchronized_host_scope='whole kernel including allocation and transfers',
                      fan='automatic; fixed-fan protocol not satisfied'))
    (OUT / 'final-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Saved', len(hashes), 'source/tool hashes')


if __name__ == '__main__':
    main()
