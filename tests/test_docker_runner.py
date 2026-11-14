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

    def test_speq_host_paths_and_comparison_image(self):
        args = self.invoke('graph_sizes', '--stage', 'plan')
        self.assertIn('looplibrarian:comparisons', args)
        self.assertIn('/var/run/docker.sock:/var/run/docker.sock', args)
        self.assertIn('LL_SPEQ_ROOT=' + str(ROOT / 'third_party/speq'), args)
        self.assertEqual(args[args.index('-w') + 1], str(ROOT))

    def test_setup_uses_installed_sources(self):
        args = self.invoke('setup', 'ggml')
        workdirs = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == '-w']
        self.assertEqual(workdirs[-1], '/opt/looplibrarian')
        self.assertEqual(args[-4:], ['looplibrarian:artifact', 'python3', 'setup.py', 'ggml'])

    def test_no_credential_mount_from_obsolete_configuration(self):
        args = self.invoke('shell', extra_env={'LL_DOCKER_SSH_DIR': '/missing/private-keys'})
        self.assertFalse(any('/root/.ssh' in arg or '/missing/private-keys' in arg for arg in args))


if __name__ == '__main__':
    unittest.main()
