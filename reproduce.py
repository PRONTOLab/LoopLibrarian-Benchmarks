#!/usr/bin/env python3
"""Run one paper evaluation, recording commands and provenance in a fresh directory."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent / 'scripts/artifact'), str(Path(__file__).resolve().parent / 'scripts/correctness')]
import argparse
import uuid
import os
from pathlib import Path
import subprocess
import sys

from common import ARTIFACT, ROOT, environment, provenance, save, setting

SUITES = ['polybench_cpu', 'polybench_gpu', 'llama', 'mfem', 'eqsat', 'eqsat_extra',
          'graph_sizes', 'ginsbach', 'kernelfarer', 'aten', 'library_inventory', 'pva', 'compilation_cost']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite', choices=SUITES)
    p.add_argument('--stage', choices=['plan', 'match', 'build', 'run', 'report', 'summarize', 'plot', 'all'], default='all')
    p.add_argument('--output', type=Path)
    p.add_argument('--cases', nargs='+', help='Explicit subset; omit for the full paper cohort')
    p.add_argument('--gpu-target', choices=['x86_64', 'jetson'], help='GPU execution platform (default: x86_64)')
    p.add_argument('--gpu-arch', help='CUDA architecture, e.g. sm_80 or sm_87')
    p.add_argument('--cohort', choices=['paper', 'all'], help='ATen selection')
    a, extra = p.parse_known_args()
    if a.cohort:
        if a.suite != 'aten':
            p.error('--cohort applies to ATen only')
        extra += ['--cohort', a.cohort]
    matching_only = a.suite == 'aten' and (a.stage == 'match' or a.cohort == 'all')
    if a.gpu_target:
        os.environ['LL_GPU_TARGET'] = a.gpu_target
    elif a.suite == 'pva':
        os.environ['LL_GPU_TARGET'] = 'jetson'
    if a.gpu_arch:
        os.environ['LL_GPU_ARCH'] = a.gpu_arch
    if a.stage in ('match', 'report', 'plot') and not (
            a.suite == 'aten' or (a.suite == 'mfem' and a.stage != 'plot')):
        p.error('match/report stages require ATen or MFEM; plot requires ATen')
    if a.stage in ('run', 'report', 'summarize', 'plot') and a.output is None:
        p.error('--output is required when resuming a built experiment')
    stamp = uuid.uuid4().hex[:12]
    out = (a.output or ROOT/'results/runs'/f'{a.suite}-{stamp}').resolve()
    directories = {'polybench_cpu': 'polybench/cpu', 'polybench_gpu': 'polybench/gpu', 'llama': 'llama', 'mfem': 'mfem', 'eqsat': 'eqsat', 'graph_sizes': 'graph_sizes', 'ginsbach': 'ginsbach', 'kernelfarer': 'kernelfarer', 'aten': 'aten', 'library_inventory': 'library_inventory'}
    directories.update(pva='pva',compilation_cost='compilation_cost')
    directories['eqsat_extra']='eqsat'
    script = ROOT/directories[a.suite]/'reproduce.py'
    if a.suite == 'eqsat_extra':script=ROOT/'eqsat/execute.py'
    stage = 'all' if a.stage == 'plan' else a.stage
    cmd = [sys.executable, str(script), '--stage', stage]
    if a.suite == 'aten':
        cmd += ['--output', str(out)]
    if a.suite == 'polybench_cpu':
        if stage == 'summarize':
            p.error('PolyBench CPU writes RESULTS.csv during --stage run')
        cmd += ['--output', str(out),
                '--cpu', str(setting('cpu', 0))]
        if setting('openblas_root'):
            cmd += ['--openblas-root', str(setting('openblas_root'))]
        if a.cases:
            cmd += ['--kernels', ','.join(a.cases)]
    elif a.cases:
        cmd += ['--cases', *a.cases]
    cmd += extra
    if a.stage == 'plan':
        print(' '.join(cmd))
        print('Output:', out)
        return 0
    fresh = stage in ('all', 'match') or (stage == 'build' and a.suite != 'aten')
    if fresh and out.exists() and any(out.iterdir()):
        p.error('Use a fresh output for match/all; resume with build/run/report/plot')
    out.mkdir(parents=True, exist_ok=True)
    import gpu_target
    gpu_suites = ('polybench_gpu', 'llama', 'mfem', 'aten', 'eqsat_extra')
    target_record = out/'gpu-target.json'
    if a.suite in gpu_suites:
        import json
        if target_record.exists():
            recorded = json.loads(target_record.read_text())
            if a.gpu_target and a.gpu_target != recorded['target']:
                p.error('The selected GPU target differs from the existing build')
            if a.gpu_arch and a.gpu_arch != recorded['architecture']:
                p.error('The selected CUDA architecture differs from the existing build')
            os.environ['LL_GPU_TARGET'] = recorded['target']
            os.environ['LL_GPU_ARCH'] = recorded['architecture']
        elif stage not in ('run', 'report', 'summarize', 'plot'):
            save(target_record, dict(target=gpu_target.name(), architecture=gpu_target.architecture()))
        if stage in ('run', 'all') and not matching_only:
            gpu_target.require_execution()
    env = environment(LL_ARTIFACT_OUTPUT=out, LL_ARTIFACT_SUITE=a.suite)
    env.setdefault('NETLIB_REF', str(Path(env['POLYGEIST_DEPENDENCY_ROOT']) / 'lapack'))
    os.environ.update(env)
    if a.suite in gpu_suites and stage in ('build', 'all') and not matching_only:
        gpu_target.validate_toolkit()
    provenance(out if stage in ('match', 'all') or (stage == 'build' and not (out/'provenance.json').exists())
               else out/('stage-'+stage))
    save(out/('command-'+stage+'.json'), cmd)
    # Keep timing campaigns serial. The drivers may parallelize compilation only.
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode

if __name__ == '__main__':
    sys.exit(main())
