"""Execute additional PolyBench recognitions, separately from the paper figures."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys

from common import ROOT, OUT, PYTHON, command, environment, save, sha

GPU = ROOT/'polybench/gpu'
sys.path.insert(0, str(GPU))
import audit
import build as gpu_build

CASES = ['fdtd-2d', 'ludcmp']


def build(cases, size):
    import os
    # Retain the views used in the EqSat coverage experiment.
    os.environ.update(environment(POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE='0'))
    shutil.copyfile(GPU/'iteration_driver.c', OUT/'iteration_driver.c')
    command(['aarch64-linux-gnu-gcc', '-O2', '-c', OUT/'iteration_driver.c',
             '-o', OUT/'iteration_driver.o'], OUT/'driver-build.log')
    save(OUT/'cases.json', cases)
    failed = []
    for name in cases:
        row = gpu_build.build(name, size=size)
        if any(row.get(k, 1) for k in ('source_build_rc', 'gpu_build_rc', 'reference_build_rc')):
            failed.append(name)
        elif not row.get('matched_calls'):
            failed.append(name)
        else:
            row['reference_sha256'] = sha(OUT/name/'reference')
            save(OUT/name/'build.json', row)
    if failed:
        raise RuntimeError('Incomplete builds: '+', '.join(failed))


def summarize(cases):
    rows = []
    for name in cases:
        folder = OUT/name
        row = dict(case=name, passed=False)
        try:
            built = json.loads((folder/'build.json').read_text())
            dims = built['dimensions']
            expected = ({key:dims['NX']*dims['NY'] for key in ('ex', 'ey', 'hz')}
                        if name == 'fdtd-2d' else {'x':dims['N']})
            reference = audit.arrays(folder/'reference.log', expected, enclosing_markers=False)
            checks = {}
            runs = {}
            for variant in ('reference', 'raised', 'raised-trace'):
                runs[variant] = json.loads((folder/(variant+'-run.json')).read_text())
                run = runs[variant]
                executable = folder/('raised' if variant.startswith('raised') else 'reference')
                if (run['rc'] or run['warmups'] != 5 or run['measured'] != 5 or
                        run['dump_count'] != 1 or not run['thermal_ok'] or
                        run['sha256'] != sha(executable)):
                    raise ValueError('Invalid execution: '+variant)
                if variant != 'reference':
                    checks[variant] = audit.compare(reference, audit.arrays(
                        folder/(variant+'.log'), expected, enclosing_markers=False))
            if sha(folder/'export/match.json') != built['proof_sha256']:
                raise ValueError('Matching proof changed after build')
            if sha(folder/'raised') != built['raised_sha256'] or sha(folder/'reference') != built['reference_sha256']:
                raise ValueError('Executable changed after build')
            timings = {key:runs['raised'][key] for key in ('compute_device_ms', 'e2e_host_ms')}
            for timing in timings.values():
                samples = timing['samples_ms']
                if len(samples) != 5 or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError('Invalid timing samples')
            reference_samples = [1000*float(x) for x in re.findall(
                r'^([0-9]+\.[0-9]+)$', (folder/'reference.log').read_text(), re.M)]
            if len(reference_samples) != 10:
                raise ValueError('Missing reference timings')
            vendor_trace = runs['raised-trace']['vendor_trace']
            row.update(dataset=built['dataset'], expected_outputs=expected,
                       emitted=built['matched_calls'], correctness=checks,
                       raised_timings=timings, reference_wall_samples_ms=reference_samples[5:],
                       vendor_trace=vendor_trace,
                       passed=all(c['passed'] for c in checks.values()) and bool(vendor_trace))
        except (OSError, KeyError, ValueError, AssertionError) as error:
            row['error'] = str(error)
        rows.append(row)
        print(name, 'PASS' if row['passed'] else 'FAIL', row.get('error', ''), flush=True)
    save(OUT/'summary.json', dict(results=rows, passed=sum(r['passed'] for r in rows),
         atol=audit.ATOL, rtol=audit.RTOL,
         scope='Additional recognition checks at the recorded dimensions; no paper speedup claim.'))
    return all(row['passed'] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['build', 'run', 'summarize', 'all'], default='all')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--size', choices=['MINI', 'SMALL', 'MEDIUM', 'LARGE', 'EXTRALARGE'], default='MINI')
    args = parser.parse_args()
    if args.stage in ('build', 'all'):
        build(args.cases, args.size)
    else:
        args.cases = json.loads((OUT/'cases.json').read_text())
    if args.stage in ('run', 'all'):
        for name in args.cases:
            for variant in ('reference', 'raised', 'raised-trace'):
                cmd = [PYTHON, GPU/'run.py']
                if variant == 'raised-trace':cmd += ['--trace']
                command(cmd+[name, 'raised' if variant == 'raised-trace' else variant],
                        OUT/'run.log', check=False)
    if args.stage in ('run', 'summarize', 'all') and not summarize(args.cases):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
