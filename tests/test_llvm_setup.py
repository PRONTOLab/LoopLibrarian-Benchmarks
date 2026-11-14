"""LLVM setup must reproduce the recorded tree without overwriting local edits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'dependency_sources/prepare_llvm.py'
spec = importlib.util.spec_from_file_location('prepare_llvm', MODULE)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class LLVMSetup(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.upstream = self.root / 'upstream'
        self.upstream.mkdir()
        self.git('init')
        self.git('config', 'user.name', 'Artifact tests')
        self.git('config', 'user.email', 'artifact@example.invalid')
        (self.upstream / 'parser.cpp').write_text('allow_result_number = false;\n')
        (self.upstream / 'other.cpp').write_text('unchanged\n')
        self.git('add', '.')
        self.git('commit', '-m', 'Initial source')
        self.base = self.git('rev-parse', 'HEAD')
        (self.upstream / 'parser.cpp').write_text('allow_result_number = true;\n')
        patch = subprocess.check_output(['git', '-C', str(self.upstream), 'diff'])
        (self.root / 'parser.patch').write_bytes(patch)
        self.git('add', '.')
        self.git('commit', '-m', 'Support result numbers')
        self.patched = self.git('rev-parse', 'HEAD')
        self.tree = self.git('rev-parse', 'HEAD^{tree}')
        self.manifest = self.root / 'llvm.json'
        self.manifest.write_text(json.dumps(dict(
            url=str(self.upstream), revision=self.base, patched_revision=self.patched,
            tree=self.tree, patch='parser.patch',
            patch_sha256=hashlib.sha256(patch).hexdigest())))
        self.target = self.root / 'target'

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.upstream), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()

    def test_fresh_checkout_and_repeat_reproduce_tree(self):
        setup.prepare(self.target, self.manifest)
        setup.prepare(self.target, self.manifest)
        self.assertEqual((self.target / 'parser.cpp').read_text(),
                         'allow_result_number = true;\n')
        self.assertEqual(setup.git(self.target, 'diff', '--exit-code', self.tree).stdout, '')

    def test_local_edits_are_preserved_and_rejected(self):
        setup.prepare(self.target, self.manifest)
        local = self.target / 'other.cpp'
        local.write_text('local work\n')
        with self.assertRaisesRegex(RuntimeError, 'local LLVM changes'):
            setup.prepare(self.target, self.manifest)
        self.assertEqual(local.read_text(), 'local work\n')

    def test_patch_checksum_is_checked_before_checkout(self):
        (self.root / 'parser.patch').write_text('incorrect patch\n')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            setup.prepare(self.target, self.manifest)
        self.assertFalse(self.target.exists())

    def test_existing_tested_commit_is_accepted(self):
        setup.prepare(self.upstream, self.manifest)
        self.assertEqual(self.git('rev-parse', 'HEAD'), self.patched)
        self.assertEqual(self.git('status', '--porcelain'), '')


if __name__ == '__main__':
    unittest.main()
