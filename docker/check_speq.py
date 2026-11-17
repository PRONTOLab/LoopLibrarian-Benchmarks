#!/usr/bin/env python3
"""Check the bundled SpEQ runtime before publishing the combined image."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/artifact'), str(ROOT/'scripts/correctness')]
from common import required, save
import speq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/opt/speq-smoke-test'))
    args = parser.parse_args()
    before = version('egglog')
    python = Path(required('speq_root'))/'venv/bin/python3'
    comparison = subprocess.check_output(
        [python, '-I', '-c', 'from importlib.metadata import version; print(version("egglog"))'],
        text=True).strip()
    if (before, comparison) != ('11.4.0', '1.0.0'):
        raise RuntimeError(f'Unexpected egglog versions: compiler={before}, SpEQ={comparison}')
    source = speq.source_path('benchmarks/gemm_ref.c')
    linear, = speq.run(args.output/'linearized', [source], func='gemm_ref')
    graph, = speq.run(args.output/'graph', [source], graph=True, func='gemm_ref')
    native, = speq.run(args.output/'native',
                      [ROOT/'polybench/sources/linear-algebra/blas/gemm/gemm.c'],
                      func='AUTO', cflags=f'-I"{ROOT}/polybench/sources/utilities" '
                                         '-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE')
    nodes = [nest['egraph_nodes'] for nest in graph['nests']
             if 'gemm' in nest.get('matched', [])]
    if (linear.get('n_match') != 1 or not nodes or max(nodes) != 370 or
            native.get('status') != 'ok' or native.get('n_match') != 0):
        raise RuntimeError(f'SpEQ smoke check differs; inspect {args.output}')
    if version('egglog') != before:
        raise RuntimeError('SpEQ changed the compiler Python environment')
    summary = dict(linearized_gemm_matches=linear['n_match'], speq_graph_nodes=max(nodes),
                   unmodified_gemm_matches=native['n_match'],
                   compiler_egglog=before, speq_egglog=comparison)
    save(args.output/'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
