"""Run the production frontend, raising recipe, and verified EqSat matcher."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/correctness')]
from ll_paths import COMPILER


def run(request):
    sys.path.insert(0, str(COMPILER/'scripts/correctness/lib_v2'))
    import recipe

    source = request['source']
    function = request['function']
    work = Path(request['work'])
    work.mkdir(parents=True, exist_ok=True)
    scripts = COMPILER/'scripts/correctness'

    def command(args):
        args = list(map(str, args))
        print('COMMAND '+shlex.join(args), flush=True)
        subprocess.run(args, check=True)

    print('[1/3] cgeist -> affine MLIR', flush=True)
    frontend = [recipe.CGEIST, source, '--function='+function,
                '--resource-dir='+recipe.RESOURCE_DIR,
                *recipe.production_arguments('frontend'),
                *shlex.split(os.environ.get('POLYGEIST_CGEIST_FLAGS', '')),
                *request['cflags']]
    affine = work/'affine.mlir'
    command([*frontend, '-o', affine])
    if not recipe.has_function(affine, function):
        affine.rename(work/'affine_explicit_missing.mlir')
        command([*frontend, '-Dstatic=', '-o', affine])
        if not recipe.has_function(affine, function):
            raise RuntimeError('cgeist did not emit requested function '+function)

    print('[2/3] affine -> Linalg -> tensor IR', flush=True)
    scope = ['--keep-dependencies'] if os.environ.get('POLYGEIST_KEEP_DEPENDENCIES', '0') != '0' else []
    command([sys.executable, scripts/'lib_v2/recipe.py', '--stage', 'raise',
             '--function', function, *scope, '--input', affine, '--output', work/'raised.mlir'])
    options = request['matching_options']
    if options.get('matching_stage') == 'linalg':
        shutil.copyfile(work/'raised.mlir', work/'linalg.mlir')
    else:
        command([sys.executable, scripts/'lib_v2/recipe.py', '--stage', 'debufferize',
                 '--function', function, *scope, '--input', work/'raised.mlir',
                 '--output', work/'linalg.mlir'])

    print('[3/3] equality saturation and specification matching', flush=True)
    rank = options.get('ranks', [0])[0]
    flags = ['--backend', 'cuda', '--region-budget', str(options.get('region_budget', 60)),
             '--max-operand-rank', str(rank)]
    if not options.get('region', True):
        flags.append('--no-region-match')
    if rank > 0:
        flags.append('--fuse-initializers')
    for name in ('split_reduction_inputs', 'split_pointwise_inputs'):
        if options.get(name, False):
            flags.append('--'+name.replace('_', '-'))
    for name in options.get('specifications') or []:
        flags.extend(['--only-kernel', name])
    command([sys.executable, scripts/'spec_match.py', work/'linalg.mlir', *flags,
             '--report', work/'match.json', '-o', work/'matched.mlir'])
    command([sys.executable, scripts/'spec_match.py', work/'matched.mlir',
             '--verify-report', work/'match.json', '--backend', 'cuda'])


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
