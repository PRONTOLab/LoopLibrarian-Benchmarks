"""The compiler and benchmark checkout can live in separate container mounts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GPUProvenance(unittest.TestCase):
    def test_changed_manifest_is_rejected_before_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = '''
import json
from pathlib import Path
import save_provenance as provenance
root = Path(__import__('sys').argv[1])
provenance.COMPILER = root / 'compiler'
provenance.OUT = root / 'output'
compiler, output = provenance.COMPILER, provenance.OUT
expected = {}
for name in ['build/bin/polygeist-opt', 'library.mlir', 'import_manifest.json']:
    path = compiler / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    expected[name] = provenance.sha(path)
proof = output / 'gemm/export/match.json'
proof.parent.mkdir(parents=True)
proof.write_text(json.dumps(dict(semantic_inputs={'import_manifest.json': expected['import_manifest.json']},
    library='library.mlir', library_sha256=expected['library.mlir'])))
(output / 'cases.json').write_text('["gemm"]')
(output / 'gemm/build.json').write_text(json.dumps(dict(proof_sha256=provenance.sha(proof),
    compiler_sha256=expected['build/bin/polygeist-opt'])))
provenance.verify_build_inputs(output, compiler)
(compiler / 'import_manifest.json').write_text('new image')
try:
    provenance.main()
except RuntimeError as error:
    assert 'import_manifest.json differs from the build' in str(error), error
else:
    raise AssertionError('Mismatched manifest accepted')
assert not (output / 'source-snapshot-final').exists()
assert not (output / 'final-provenance.json').exists()
'''
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
                       ['polybench/gpu', 'scripts/artifact', 'scripts/correctness']))
            subprocess.check_call([sys.executable, '-c', code, tmp], env=env,
                                  stdout=subprocess.DEVNULL)

    def test_native_build_with_dependencies_outside_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = '''
import json
from pathlib import Path
import build_native as native
root = Path(__import__('sys').argv[1])
native.DEPENDENCIES = root / 'installed/dependencies'
native.UP = native.DEPENDENCIES / 'polybenchGpu'
native.OUT = root / 'output'
source = native.UP / 'CUDA/2MM/2mm.cu'
source.parent.mkdir(parents=True)
source.write_text('__global__ void mm2_kernel() {}')
source.with_suffix('.cuh').write_text('#define DIM_THREAD_BLOCK_X 32')
(native.OUT / '2mm').mkdir(parents=True)
native.command = lambda *args, **kwargs: 7
rc = native.build('2mm')
record = json.loads((native.OUT / '2mm/native-build.json').read_text())
assert rc == 7 and record['build_rc'] == 7, record
assert record['upstream_source'] == 'third_party/polybenchGpu/CUDA/2MM/2mm.cu', record
assert 'executable_sha256' not in record
'''
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
                       ['polybench/gpu', 'scripts/artifact', 'scripts/correctness']))
            subprocess.check_call([sys.executable, '-c', code, tmp], env=env,
                                  stdout=subprocess.DEVNULL)

    def test_separate_roots_preserve_compiler_hash_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compiler = root / 'installed/compiler'
            benchmark = root / 'mounted/benchmarks'
            for directory, content in [(compiler, 'compiler implementation'),
                                       (benchmark, 'benchmark wrapper')]:
                path = directory / 'scripts/correctness/build_jetson.sh'
                path.parent.mkdir(parents=True)
                path.write_text(content)
            destination = root / 'snapshot'
            code = '''
import json
from pathlib import Path
from save_provenance import snapshot
compiler, benchmark, destination = map(Path, __import__('sys').argv[1:])
name = 'scripts/correctness/build_jetson.sh'
hashes = snapshot([compiler / name], compiler, destination / 'compiler')
hashes.update(snapshot([benchmark / name], benchmark, destination / 'benchmarks', 'benchmarks/'))
print(json.dumps(hashes))
'''
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
                       ['polybench/gpu', 'scripts/artifact', 'scripts/correctness']))
            output = subprocess.check_output([sys.executable, '-c', code, str(compiler),
                                               str(benchmark), str(destination)], env=env, text=True)
            hashes = json.loads(output)
            key = 'scripts/correctness/build_jetson.sh'
            self.assertEqual(hashes[key], hashlib.sha256(b'compiler implementation').hexdigest())
            self.assertEqual(hashes['benchmarks/' + key],
                             hashlib.sha256(b'benchmark wrapper').hexdigest())
            self.assertEqual((destination / 'compiler' / key).read_text(), 'compiler implementation')
            self.assertEqual((destination / 'benchmarks' / key).read_text(), 'benchmark wrapper')


if __name__ == '__main__':
    unittest.main()
