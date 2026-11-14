"""Public header setup preserves local files and verifies downloaded archives."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cuda_headers', ROOT / 'dependency_sources/prepare_cuda_headers.py')
headers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(headers)


class CUDAHeaders(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.archive = self.root / 'public.tar.xz'
        with tarfile.open(self.archive, 'w:xz') as archive:
            for name, data in [('include/runtime.h', b'public header\n'),
                               ('LICENSE', b'license\n'), ('lib/runtime.so', b'binary')]:
                info = tarfile.TarInfo('package/' + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        self.manifest = self.root / 'packages.json'
        self.manifest.write_text(json.dumps({'packages': [dict(
            component='runtime', directory='cudart', url=self.archive.as_uri(),
            sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest())]}))
        self.destination = self.root / 'headers'

    def test_headers_and_license_only_and_repeat(self):
        headers.prepare(self.destination, self.manifest)
        headers.prepare(self.destination, self.manifest)
        self.assertEqual((self.destination / 'cudart/include/runtime.h').read_bytes(), b'public header\n')
        self.assertTrue((self.destination / 'cudart/LICENSE').is_file())
        self.assertFalse((self.destination / 'cudart/lib').exists())

    def test_local_edits_are_preserved(self):
        headers.prepare(self.destination, self.manifest)
        header = self.destination / 'cudart/include/runtime.h'
        header.write_text('local edit\n')
        with self.assertRaisesRegex(RuntimeError, 'existing header differs'):
            headers.prepare(self.destination, self.manifest)
        self.assertEqual(header.read_text(), 'local edit\n')

    def test_bad_checksum_is_rejected_before_writing(self):
        self.archive.write_bytes(b'corrupted download')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            headers.prepare(self.destination, self.manifest)
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
