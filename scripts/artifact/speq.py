"""Execute the real SpEQ artifact with its isolated LLVM/Python toolchain."""
import json
from pathlib import Path
from common import ROOT, command, environment, required, save, setting, sha


def source_path(relative):
    return Path(required('speq_root')).resolve()/'lleq-artifact'/relative


def run(output, files, *, graph=False, func=None, cflags=''):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(required('speq_root')).resolve()
    source = source_path('')
    script = ROOT/('graph_sizes/speq_capture.py' if graph else 'eqsat/speq_match.py')
    python = root/'venv/bin/python3'
    if not python.is_file() or not (root/'runtime.json').is_file():
        raise RuntimeError('SpEQ runtime missing: pull the current combined image, '
                           'or run python3 setup.py speq for a native installation')
    # Ignore the compiler's PYTHONPATH and user site packages: SpEQ needs egglog 1.
    args = [python, '-I', script, '--out', output/'results.json']
    if func: args += ['--func', func]
    if cflags: args += ['--cflags', cflags]
    args += files
    command(args, output/'run.log', timeout=int(setting('speq_timeout', 1800)),
            env=environment(LL_SPEQ_ROOT=root), cwd=source)
    save(output/'runtime.json', json.loads((root/'runtime.json').read_text()))
    save(output/'inputs.json', dict(driver_sha256=sha(script),
         source_manifest=json.loads((root/'source-files.json').read_text()),
         artifact_hashes={str(p.relative_to(source)):sha(p) for p in source.glob('*.py')}))
    return json.loads((output/'results.json').read_text())
