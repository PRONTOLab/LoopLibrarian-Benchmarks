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


def main():
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
