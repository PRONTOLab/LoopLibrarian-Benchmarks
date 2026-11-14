"""Local staging, process cleanup, and target architecture checks."""
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/artifact'))
import local_execution as execution


class LocalExecution(unittest.TestCase):
    def test_run_is_local_even_with_old_transport_configuration(self):
        with patch.dict(os.environ, {'LL_SSH_HOST': 'unused.invalid',
                                     'POLYGEIST_REMOTE_TRANSPORT': '/missing/adapter.py'}):
            rc, output = execution.run('printf local; exit 7')
        self.assertEqual((rc, output), (7, 'local'))

    def test_staging_handles_spaces_and_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input file'
            target = Path(directory) / 'nested dir' / 'output file'
            source.write_bytes(b'benchmark')
            source.chmod(0o755)
            self.assertEqual(execution.stage(source, target), 0)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertTrue(os.access(target, os.X_OK))
            self.assertEqual(execution.stage(target, target), 0)

    def test_timeout_stops_child_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'must-not-be-written'
            command = '(sleep 0.3; touch ' + shlex.quote(str(marker)) + ') & wait'
            rc, _ = execution.run(command, timeout=0.05)
            self.assertEqual(rc, 124)
            execution.run('sleep 0.4')
            self.assertFalse(marker.exists())

    def test_cross_architecture_execution_is_rejected(self):
        with patch.object(execution.platform, 'machine', return_value='x86_64'):
            with self.assertRaisesRegex(RuntimeError, 'AArch64'):
                execution.require_aarch64()
        with patch.object(execution.platform, 'machine', return_value='aarch64'):
            execution.require_aarch64()


if __name__ == '__main__':
    unittest.main()
