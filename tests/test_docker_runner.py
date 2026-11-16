"""Check host mounts used by the container and by SpEQ's sibling container."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DockerRunner(unittest.TestCase):
    def invoke(self, *args, extra_env=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'docker'
            stub.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith('LL_DOCKER_')}
            env.update(extra_env or {})
            env['PATH'] = str(root) + os.pathsep + env['PATH']
            return json.loads(subprocess.check_output(['bash', str(ROOT / 'docker/run.sh'), *args],
                                                     env=env, text=True))

    def test_results_and_builds_survive_normal_run(self):
        args = self.invoke('polybench_cpu', '--stage', 'plan')
        self.assertIn(str(ROOT) + ':' + str(ROOT), args)
        self.assertIn('looplibrarian-builds:/opt/looplibrarian/build', args)
        self.assertEqual(args[-6:], ['looplibrarian:artifact', 'python3', 'reproduce.py',
                                    'polybench_cpu', '--stage', 'plan'])
        self.assertNotIn('/var/run/docker.sock:/var/run/docker.sock', args)

    def test_speq_host_paths_and_docker_socket(self):
        args = self.invoke('graph_sizes', '--stage', 'plan')
        self.assertIn('looplibrarian:artifact', args)
        self.assertIn('/var/run/docker.sock:/var/run/docker.sock', args)
        self.assertIn('LL_SPEQ_ROOT=' + str(ROOT / 'third_party/speq'), args)
        self.assertEqual(args[args.index('-w') + 1], str(ROOT))

    def test_setup_uses_installed_sources(self):
        args = self.invoke('setup', 'ggml')
        workdirs = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == '-w']
        self.assertEqual(workdirs[-1], '/opt/looplibrarian')
        self.assertEqual(args[-4:], ['looplibrarian:artifact', 'python3', 'setup.py', 'ggml'])

    def test_native_gpu_execution_exposes_gpu(self):
        args = self.invoke('polybench_gpu', '--stage', 'run')
        self.assertIn('looplibrarian:artifact', args)
        self.assertIn('--gpus', args)
        self.assertIn('memlock=-1', args)
        self.assertNotIn('looplibrarian-builds:/opt/looplibrarian/build', args)

    def test_matching_only_does_not_require_gpu(self):
        args = self.invoke('aten', '--cohort', 'all', '--stage', 'match')
        self.assertIn('looplibrarian:artifact', args)
        self.assertNotIn('--gpus', args)

    def test_full_aten_execution_exposes_gpu_unless_building(self):
        for stage in ('all', 'run', 'build'):
            with self.subTest(stage=stage):
                args = self.invoke('aten', '--cohort', 'all', '--stage', stage)
                self.assertIn('looplibrarian:artifact', args)
                self.assertEqual('--gpus' in args, stage != 'build')

    def test_custom_gpu_image_keeps_its_prebuilt_baseline(self):
        args = self.invoke('llama', '--stage', 'build',
                           extra_env={'LL_DOCKER_IMAGE': 'registry.example/looplibrarian:release'})
        self.assertIn('registry.example/looplibrarian:release', args)
        self.assertNotIn('looplibrarian-builds:/opt/looplibrarian/build', args)

    def test_output_directory_does_not_change_execution_stage(self):
        args = self.invoke('llama', '--output', 'build')
        self.assertIn('--gpus', args)

    def test_build_and_jetson_do_not_require_local_gpu(self):
        for options in [('aten', '--stage', 'build'),
                        ('compilation_cost', '--stage', 'all'),
                        ('llama', '--gpu-target', 'jetson', '--stage', 'build')]:
            args = self.invoke(*options)
            self.assertNotIn('--gpus', args)
        self.assertIn('looplibrarian:artifact', args)
        self.assertIn('LL_GPU_TARGET=jetson', args)

    def test_no_credential_mount_from_obsolete_configuration(self):
        args = self.invoke('shell', extra_env={'LL_DOCKER_SSH_DIR': '/missing/private-keys'})
        self.assertFalse(any('/root/.ssh' in arg or '/missing/private-keys' in arg for arg in args))

    def test_installed_pva_dependencies_are_read_only_mounts(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = {'PVA_OPERATOR_INCLUDE': 'operator', 'NVCV_INCLUDE': 'nvcv',
                        'CUPVA_SDK_ROOT': 'cupva', 'PVA_LIB_STAGE': 'lib'}
            env = {}
            for variable, name in settings.items():
                path = Path(directory)/name
                path.mkdir()
                env['LL_DOCKER_'+variable] = str(path)
            args = self.invoke('pva', '--stage', 'build', extra_env=env)
            for variable, name in settings.items():
                self.assertIn(env['LL_DOCKER_'+variable]+':/pva/'+name+':ro', args)
                self.assertIn(variable+'=/pva/'+name, args)


if __name__ == '__main__':
    unittest.main()
