"""Relocated execution keeps build evidence and does not require Git or a compiler."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('execution_bundle', ROOT/'target.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


class ExecutionBundle(unittest.TestCase):
    def fixture(self, root):
        build = root/'original-build'
        (build/'fixture').mkdir(parents=True)
        binary = build/'fixture/program'
        binary.write_bytes(b'fixture binary')
        binary.chmod(0o755)
        target.save(build/'fixture/build.json', dict(executable_sha256=target.sha(binary),
                    emitted=['reference_spec'], original_path=str(binary)))
        target.save(build/'provenance.json', dict(compiler_commit='original-compiler',
                    git_head='original-artifact', tools={'cgeist': 'original-hash'}))
        return build

    def test_relocation_preserves_proofs_and_runs_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            original = (build/'fixture/build.json').read_bytes()
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            moved = base/'different directory'/'relocated bundle'
            moved.parent.mkdir()
            shutil.move(bundle, moved)
            shutil.rmtree(build)
            manifest = target.verify(moved)
            self.assertFalse((moved/'.git').exists())
            self.assertFalse((moved/'third_party/Polygeist').exists())
            self.assertEqual((moved/'builds/mfem/fixture/build.json').read_bytes(), original)
            self.assertEqual(manifest['suites']['mfem']['build_provenance']['compiler_commit'],
                             'original-compiler')
            subprocess.run([sys.executable, str(moved/'target.py'), 'run', '--bundle', str(moved),
                            '--verify-only'], check=True, capture_output=True)
            # Execute the actual shared provenance writer in a fresh interpreter,
            # with the original build directory deleted and no compiler checkout.
            env = dict(os.environ, PYTHONPATH=str(moved/'scripts/artifact'),
                       LL_EXECUTION_BUNDLE=str(moved/'execution-bundle.json'), LL_ARTIFACT_SUITE='mfem',
                       LL_ARTIFACT_CONFIG=str(moved/'absent-config.json'))
            subprocess.run([sys.executable, '-c', 'from common import provenance; provenance("new-run")'],
                           cwd=moved, env=env, check=True, capture_output=True)
            result = target.read(moved/'new-run/provenance.json')
            self.assertEqual(result['build_provenance']['compiler_commit'], 'original-compiler')
            self.assertEqual(result['bundle_sha256'], target.sha(moved/'execution-bundle.json'))

    def test_stale_binaries_and_modified_bundle_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            (bundle/'builds/mfem/fixture/program').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed bundle file'):
                target.verify(bundle)
            (build/'fixture/program').write_bytes(b'stale')
            with self.assertRaisesRegex(ValueError, 'differs from build'):
                target.pack(base/'stale', {'mfem': build})
            self.assertFalse((base/'stale').exists())

    def test_runtime_failure_is_retained_with_unchanged_build_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            uname = type('Machine', (), {'machine': 'aarch64'})()
            def execute(command, **kwargs):
                if command[0] == 'ldd':
                    return subprocess.CompletedProcess(command, 0, 'libc.so.6 => /lib/libc.so.6', '')
                return subprocess.CompletedProcess(command, 7)
            with patch.object(target.os, 'uname', return_value=uname), patch.object(target.subprocess, 'run', side_effect=execute):
                self.assertEqual(target.run(bundle), 1)
            self.assertEqual(target.read(bundle/'runs/execution.json')['mfem']['returncode'], 7)
            self.assertEqual((bundle/'runs/mfem/fixture/program').read_bytes(), b'fixture binary')
            target.verify(bundle)  # Runtime files cannot overwrite immutable build inputs.

    def test_remote_retrieves_results_after_nonzero_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 9 if command[0] == 'ssh' and 'target.py' in command[-1] else 0)
            with patch.object(target.subprocess, 'run', side_effect=execute):
                rc = target.remote(bundle, 'reviewer@orin', '/tmp/test bundle', base/'collected',
                                   jump_host='gateway', ssh_options=['BatchMode=no'])
            self.assertEqual(rc, 9)
            self.assertEqual(calls[-1][0], 'scp')
            self.assertIn('-J', calls[-1])
            self.assertLess(calls[0].index('BatchMode=no'), calls[0].index('BatchMode=yes'))
            self.assertEqual(target.read(base/'collected/transport.json')['returncode'], 9)

    def test_pva_relocation_retains_matching_evidence_and_selected_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            proof = build/'fixture/proof.json'
            target.save(proof, {'emitted': ['reference_spec']})
            (build/'fixture/matched.mlir').write_text('module {}')
            target.save(build/'fixture/raise.json', {'raised': True})
            record = target.read(build/'fixture/build.json')
            record['proof_sha256'] = target.sha(proof)
            target.save(build/'fixture/build.json', record)
            bundle = base/'bundle'
            target.pack(bundle, {'pva': build})
            self.assertEqual(target.sha(bundle/'builds/pva/fixture/proof.json'), record['proof_sha256'])
            self.assertTrue((bundle/'pva/reproduce.py').is_file())
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, '', '')
            uname = type('Machine', (), {'machine': 'aarch64'})()
            with patch.object(target.os, 'uname', return_value=uname), patch.object(target.subprocess, 'run', side_effect=execute):
                self.assertEqual(target.run(bundle), 0)
            self.assertEqual(calls[-1][-2:], ['--cases', 'fixture'])
            proof.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'proof differs'):
                target.pack(base/'changed-proof', {'pva': build})


if __name__ == '__main__':
    unittest.main()
