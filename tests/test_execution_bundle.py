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
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for name in list(os.environ):
            if name.startswith(('LL_', 'POLYGEIST_GPU_')) or name == 'LD_LIBRARY_PATH':
                os.environ.pop(name)

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
            self.assertTrue((moved/'scripts/artifact/gpu_target.py').is_file())
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
            uname = type('Machine', (), {'machine': 'x86_64'})()
            def execute(command, **kwargs):
                if command[0] == 'ldd':
                    return subprocess.CompletedProcess(command, 0, 'libc.so.6 => /lib/libc.so.6', '')
                return subprocess.CompletedProcess(command, 7)
            with patch.object(target.os, 'uname', return_value=uname), patch.object(target.subprocess, 'run', side_effect=execute):
                self.assertEqual(target.run(bundle), 1)
            config = target.read(bundle/'runs/target-config.json')
            self.assertEqual(config['gpu_target'], 'x86_64')
            self.assertIn('/usr/lib/x86_64-linux-gnu', config['library_path'])
            self.assertNotIn('aarch64', config['library_path'])
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
                rc = target.remote(bundle, 'reviewer@gpu', '/tmp/test bundle', base/'collected',
                                   jump_host='gateway', ssh_options=['BatchMode=no'])
            self.assertEqual(rc, 9)
            self.assertEqual(calls[-1][0], 'scp')
            self.assertIn('-J', calls[-1])
            self.assertLess(calls[0].index('BatchMode=no'), calls[0].index('BatchMode=yes'))
            self.assertTrue(any('--gpu-target x86_64' in call[-1] for call in calls if call[0] == 'ssh'))
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
                self.assertEqual(target.run(bundle, gpu_target='jetson'), 0)
            self.assertEqual(target.read(bundle/'runs/target-config.json')['gpu_target'], 'jetson')
            self.assertEqual(calls[-1][-2:], ['--cases', 'fixture'])
            proof.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'proof differs'):
                target.pack(base/'changed-proof', {'pva': build})

    def test_jetson_is_explicit_and_saved_architecture_survives_packaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            target.save(build/'gpu-target.json', dict(target='jetson', architecture='sm_87'))
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            self.assertEqual((bundle/'builds/mfem/gpu-target.json').read_bytes(),
                             (build/'gpu-target.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'select --gpu-target jetson'):
                target.run(bundle)
            with patch.object(target.subprocess, 'run') as execute:
                with self.assertRaisesRegex(ValueError, 'select --gpu-target jetson'):
                    target.remote(bundle, 'reviewer@orin', '/tmp/bundle', base/'results')
                execute.assert_not_called()
            uname = type('Machine', (), {'machine': 'aarch64'})()
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, '', '')
            with patch.object(target.os, 'uname', return_value=uname), patch.object(target.subprocess, 'run', side_effect=execute):
                self.assertEqual(target.run(bundle, gpu_target='jetson'), 0)
            config = target.read(bundle/'runs/target-config.json')
            self.assertIn('/usr/lib/aarch64-linux-gnu', config['library_path'])
            self.assertEqual(target.read(bundle/'runs/mfem/gpu-target.json')['architecture'], 'sm_87')
            self.assertEqual(calls[-1][calls[-1].index('--gpu-target') + 1], 'jetson')

    def test_x86_bundle_is_rejected_on_arm_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            uname = type('Machine', (), {'machine': 'aarch64'})()
            with patch.object(target.os, 'uname', return_value=uname), patch.object(target.subprocess, 'run') as execute:
                with self.assertRaisesRegex(ValueError, 'x86_64 executables cannot run on aarch64'):
                    target.run(bundle)
                execute.assert_not_called()
            self.assertFalse((bundle/'runs').exists())

    def test_jetson_bundle_is_rejected_on_x86_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            target.save(build/'gpu-target.json', dict(target='jetson', architecture='sm_87'))
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            uname = type('Machine', (), {'machine': 'x86_64'})()
            with patch.object(target.os, 'uname', return_value=uname), patch.object(target.subprocess, 'run') as execute:
                with self.assertRaisesRegex(ValueError, 'jetson executables cannot run on x86_64'):
                    target.run(bundle, gpu_target='jetson')
                execute.assert_not_called()
            self.assertFalse((bundle/'runs').exists())

    def test_legacy_arm_binary_cannot_be_assumed_to_be_x86(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            binary = build/'fixture/program'
            header = bytearray(20)
            header[:6] = b'\x7fELF\x02\x01'
            header[18:20] = (183).to_bytes(2, 'little')
            binary.write_bytes(header)
            target.save(build/'fixture/build.json', dict(executable_sha256=target.sha(binary)))
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            with patch.object(target.subprocess, 'run') as execute:
                with self.assertRaisesRegex(ValueError, 'Executable architecture does not match x86_64'):
                    target.run(bundle)
                execute.assert_not_called()

    def test_cuda_architecture_conflict_is_rejected_before_execution_or_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            target.save(build/'gpu-target.json', dict(target='x86_64', architecture='sm_86'))
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            with patch.object(target.subprocess, 'run') as execute:
                with self.assertRaisesRegex(ValueError, 'saved CUDA architecture is sm_86'):
                    target.run(bundle, gpu_arch='sm_90')
                with self.assertRaisesRegex(ValueError, 'saved CUDA architecture is sm_86'):
                    target.remote(bundle, 'reviewer@gpu', '/tmp/bundle', base/'results', gpu_arch='sm_90')
                execute.assert_not_called()

    def test_cli_defaults_to_x86_and_forwards_explicit_jetson(self):
        for action in ('run', 'remote'):
            for options, expected in (([], ('x86_64', None)),
                                      (['--gpu-target', 'jetson', '--gpu-arch', 'sm_87'], ('jetson', 'sm_87'))):
                args = ['target.py', action, '--bundle', '/tmp/bundle', *options]
                if action == 'remote':
                    args += ['--host', 'reviewer@gpu', '--results', '/tmp/results']
                with self.subTest(action=action, options=options), patch.object(sys, 'argv', args), \
                        patch.object(target, action, return_value=0) as execute:
                    self.assertEqual(target.main(), 0)
                    self.assertEqual(execute.call_args.args[-2:], expected)

    def test_remote_jetson_forwards_settings_and_reports_retrieval_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            target.save(build/'gpu-target.json', dict(target='jetson', architecture='sm_87'))
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            for run_rc in (0, 9):
                calls = []
                def execute(command, **kwargs):
                    calls.append(command)
                    rc = 4 if command[0] == 'scp' else run_rc if 'target.py' in command[-1] else 0
                    return subprocess.CompletedProcess(command, rc)
                results = base/f'collected-{run_rc}'
                with patch.object(target.subprocess, 'run', side_effect=execute):
                    self.assertEqual(target.remote(bundle, 'reviewer@orin', '/tmp/bundle', results,
                                     gpu_target='jetson', gpu_arch='sm_87'), run_rc or 4)
                self.assertIn('--gpu-target jetson --gpu-arch sm_87', calls[-2][-1])
                self.assertEqual(calls[-1][0], 'scp')
                report = target.read(results/'transport.json')
                self.assertEqual((report['returncode'], report['retrieval_returncode']), (run_rc, 4))

    def test_runtime_configuration_and_bundle_ggml_are_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            bundle = base/'bundle'
            target.pack(bundle, {'mfem': build})
            # Stand in for the library directory normally packaged for Llama.
            ggml = bundle/'runtime/ggml'
            ggml.mkdir(parents=True)
            config_path = base/'runtime-config.json'
            target.save(config_path, dict(library_path='/runtime/cuda:/runtime/torch',
                        environment={'ATEN_NATIVE_TORCH_ROOT': '/runtime/torch',
                                     'POLYGEIST_GGML_BUILD': '/old/ggml'}))
            calls = []
            def execute(command, **kwargs):
                calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0, '', '')
            uname = type('Machine', (), {'machine': 'x86_64'})()
            with patch.dict(os.environ, LL_ARTIFACT_CONFIG=str(config_path), LL_GPU_TARGET='jetson'), \
                    patch.object(target.os, 'uname', return_value=uname), \
                    patch.object(target.subprocess, 'run', side_effect=execute):
                self.assertEqual(target.run(bundle), 0)
            config = target.read(bundle/'runs/target-config.json')
            self.assertEqual(config['library_path'], str(ggml/'ggml/src') + ':/runtime/cuda:/runtime/torch')
            self.assertEqual(config['environment']['POLYGEIST_GGML_BUILD'], str(ggml))
            self.assertEqual(config['environment']['ATEN_NATIVE_TORCH_ROOT'], '/runtime/torch')
            self.assertEqual(calls[-1][1]['env']['LL_GPU_TARGET'], 'x86_64')
            self.assertEqual(calls[0][1]['env']['LD_LIBRARY_PATH'], config['library_path'])

    def test_runtime_library_overrides_reach_dependency_check_and_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            config_path = base/'config.json'
            target.save(config_path, dict(library_path='/configured/lib',
                        environment={'LL_LIBRARY_PATH': '/stale/lib'}))
            uname = type('Machine', (), {'machine': 'x86_64'})()
            for explicit, expected in ((None, '/environment/lib'), ('/explicit/lib', '/explicit/lib')):
                bundle = base/('explicit' if explicit else 'environment')
                target.pack(bundle, {'mfem': build})
                with patch.dict(os.environ, LL_ARTIFACT_CONFIG=str(config_path), LL_LIBRARY_PATH='/environment/lib'), \
                        patch.object(target.os, 'uname', return_value=uname), \
                        patch.object(target.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as execute:
                    self.assertEqual(target.run(bundle, library_path=explicit), 0)
                config = target.read(bundle/'runs/target-config.json')
                self.assertEqual(config['library_path'], expected)
                self.assertEqual(config['environment']['LL_LIBRARY_PATH'], expected)
                self.assertEqual(execute.call_args_list[0].kwargs['env']['LD_LIBRARY_PATH'], expected)
                self.assertEqual(execute.call_args_list[-1].kwargs['env']['LL_LIBRARY_PATH'], expected)

    @unittest.skipUnless(os.uname().machine.lower() in ('x86_64', 'amd64'), 'Native x86 runner integration')
    def test_real_runner_restores_saved_settings_after_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = self.fixture(base)
            saved = b'{ "target": "x86_64", "architecture": "sm_86" }\n'
            (build/'gpu-target.json').write_bytes(saved)
            bundle = base/'bundle'
            manifest = target.pack(bundle, {'mfem': build})
            # A probe replaces only the GPU workload. The packaged target.py,
            # reproduce.py, configuration and provenance paths execute normally.
            probe = bundle/'mfem/reproduce.py'
            probe.write_text('''import os
import gpu_target
from common import OUT, save
save(OUT/'observed.json', dict(target=gpu_target.name(), architecture=gpu_target.architecture(),
     runtime_target=os.environ['POLYGEIST_GPU_TARGET'], runtime_arch=os.environ['POLYGEIST_GPU_ARCH'],
     libraries=gpu_target.library_path()))
''')
            manifest['files']['mfem/reproduce.py'] = target.sha(probe)
            manifest['runtime_sources']['mfem/reproduce.py'] = target.sha(probe)
            target.save(bundle/'execution-bundle.json', manifest)
            config_path = base/'runtime-config.json'
            target.save(config_path, dict(gpu_target='jetson', gpu_arch='sm_90',
                library_path='/runtime/lib', environment=dict(LL_GPU_TARGET='jetson',
                POLYGEIST_GPU_TARGET='jetson', LL_GPU_ARCH='sm_90', POLYGEIST_GPU_ARCH='sm_90')))
            shutil.rmtree(build)
            real_run = subprocess.run
            def execute(command, **kwargs):
                if command[0] == 'ldd':
                    return subprocess.CompletedProcess(command, 0, '', '')
                return real_run(command, **kwargs)
            with patch.dict(os.environ, LL_ARTIFACT_CONFIG=str(config_path), LL_GPU_ARCH='sm_75'), \
                    patch.object(target.subprocess, 'run', side_effect=execute):
                self.assertEqual(target.run(bundle), 0, (bundle/'runs/mfem.log').read_text())
            self.assertEqual(target.read(bundle/'runs/mfem/observed.json'), dict(target='x86_64',
                architecture='sm_86', runtime_target='x86_64', runtime_arch='sm_86', libraries='/runtime/lib'))
            self.assertEqual((bundle/'builds/mfem/gpu-target.json').read_bytes(), saved)
            self.assertEqual((bundle/'runs/mfem/gpu-target.json').read_bytes(), saved)
            self.assertEqual(target.read(bundle/'runs/mfem/stage-run/provenance.json')['build_provenance']['compiler_commit'],
                             'original-compiler')


if __name__ == '__main__':
    unittest.main()
