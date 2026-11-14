"""SpEQ imports only required public source files from its large archive."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('artifact_speq_setup', ROOT / 'setup.py')
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class SpeqSetup(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / 'artifact.tar.gz'
        self.target = self.root / 'installed'

    def make_archive(self, *, omit=None, source_link=None):
        with tarfile.open(self.archive, 'w:gz') as tar:
            link = tarfile.TarInfo('lleq-artifact/oneapi/unused-link')
            link.type = tarfile.SYMTYPE
            link.linkname = '/opt/intel/oneapi/unused'
            tar.addfile(link)
            for name in setup.SPEQ_SOURCE_FILES:
                if name == omit:
                    continue
                member = tarfile.TarInfo('lleq-artifact/' + name)
                if name == source_link:
                    member.type = tarfile.SYMTYPE
                    member.linkname = '/outside'
                    tar.addfile(member)
                else:
                    body = (name + '\n').encode()
                    member.size = len(body)
                    tar.addfile(member, io.BytesIO(body))
        digest = hashlib.md5(self.archive.read_bytes()).hexdigest()
        mock = patch.object(setup, 'SPEQ_ARCHIVE_MD5', digest)
        mock.start()
        self.addCleanup(mock.stop)

    def test_import_skips_bundled_tools_and_validates_cached_sources(self):
        self.make_archive()
        setup.prepare_speq_sources(self.archive, self.target)
        source = self.target / 'lleq-artifact'
        self.assertEqual({str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()},
                         set(setup.SPEQ_SOURCE_FILES))
        self.assertFalse((source / 'oneapi').exists())
        self.archive.unlink()
        setup.prepare_speq_sources(self.archive, self.target)
        (source / 'parseIR.py').write_text('modified')
        with self.assertRaisesRegex(RuntimeError, 'changed or is missing'):
            setup.prepare_speq_sources(self.archive, self.target)

    def test_required_source_link_is_rejected(self):
        self.make_archive(source_link='parseIR.py')
        with self.assertRaisesRegex(RuntimeError, 'regular file'):
            setup.prepare_speq_sources(self.archive, self.target)
        self.assertFalse((self.target / 'source-files.json').exists())

    def test_incomplete_import_is_not_recorded_as_ready(self):
        self.make_archive(omit='analysis/gemm_ref.ll')
        with self.assertRaisesRegex(RuntimeError, 'missing sources'):
            setup.prepare_speq_sources(self.archive, self.target)
        self.assertFalse((self.target / 'lleq-artifact').exists())

    def test_existing_changes_are_preserved(self):
        self.make_archive()
        source = self.target / 'lleq-artifact'
        source.mkdir(parents=True)
        (source / 'parseIR.py').write_text('local edits')
        with self.assertRaisesRegex(RuntimeError, 'left untouched'):
            setup.prepare_speq_sources(self.archive, self.target)
        self.assertEqual((source / 'parseIR.py').read_text(), 'local edits')
        self.assertFalse((source / 'driver.py').exists())

    def test_invalid_archive_checksum_is_rejected(self):
        self.make_archive()
        self.archive.write_bytes(self.archive.read_bytes() + b'changed')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            setup.prepare_speq_sources(self.archive, self.target)


if __name__ == '__main__':
    unittest.main()
