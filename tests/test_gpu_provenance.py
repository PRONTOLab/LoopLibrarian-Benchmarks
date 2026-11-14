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
