"""Exercise release scripts against Git repositories and a recorded Docker CLI."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DOCKER = '''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
record = {'args': args}
if args[0] == 'build':
    root = pathlib.Path(args[-1])
    record['context'] = {str(p.relative_to(root)): p.read_text()
                         for p in root.rglob('*') if p.is_file()}
with open(os.environ['DOCKER_CALLS'], 'a') as stream:
    stream.write(json.dumps(record) + '\\n')
if args[:2] == ['image', 'inspect']:
    image = args[2]
    if os.environ.get('MISSING_IMAGE') == image:
        sys.exit(1)
    print('outdated' if os.environ.get('STALE_IMAGE') == image else os.environ['EXPECTED_REVISION'])
elif args[0] == 'push' and os.environ.get('FAIL_PUSH'):
    sys.exit(1)
'''


class DockerRelease(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'checkout'
        self.root.mkdir()
        (self.root / 'docker').mkdir()
        for name in ('build.sh', 'publish.sh'):
            shutil.copy2(ROOT / 'docker' / name, self.root / 'docker' / name)
        (self.root / 'Dockerfile').write_text('FROM scratch\n')
        (self.root / '.dockerignore').write_text('results/\n')
        self.git('init', '-q')
        self.git('add', '.')
        self.git('update-index', '--add', '--cacheinfo',
                 '160000,' + '1' * 40 + ',third_party/Polygeist')
        (self.root / 'third_party/Polygeist').mkdir(parents=True)
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-qm', 'Fixture')
        self.revision = self.git('rev-parse', 'HEAD').stdout.strip()
        self.bin = Path(self.temporary.name) / 'bin'
        self.bin.mkdir()
        (self.bin / 'docker').write_text(DOCKER)
        (self.bin / 'docker').chmod(0o755)
        self.log = Path(self.temporary.name) / 'docker-calls.jsonl'
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('LL_DOCKER_', 'GIT_'))}
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env['PATH'],
                        DOCKER_CALLS=str(self.log), EXPECTED_REVISION=self.revision)

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, text=True,
                              capture_output=True, check=True)

    def invoke(self, script, *args, **env):
        result = subprocess.run(['bash', str(self.root / 'docker' / script), *args],
                                cwd=self.root, env=dict(self.env, **env),
                                text=True, capture_output=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_build_uses_committed_context_and_removes_temporary_files(self):
        (self.root / 'local-secret.txt').write_text('not part of the artifact')
        result, calls = self.invoke('build.sh', '--comparisons')
        self.assertEqual(result.returncode, 0, result.stderr)
        call, = calls
        self.assertEqual(call['context']['Dockerfile'], 'FROM scratch\n')
        self.assertNotIn('local-secret.txt', call['context'])
        self.assertNotIn('.git/config', call['context'])
        self.assertIn('ARTIFACT_REVISION=' + self.revision, call['args'])
        self.assertIn('POLYGEIST_REVISION=' + '1' * 40, call['args'])
        self.assertIn('looplibrarian:comparisons', call['args'])
        self.assertFalse(Path(call['args'][-1]).exists())

    def test_release_rejects_staged_and_unstaged_changes(self):
        (self.root / 'Dockerfile').write_text('FROM changed\n')
        for staged in (False, True):
            if staged:
                self.git('add', 'Dockerfile')
            for script in ('build.sh', 'publish.sh'):
                with self.subTest(staged=staged, script=script):
                    result, calls = self.invoke(script)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(calls, [])

    def test_publication_checks_both_images_before_pushing(self):
        result, calls = self.invoke('publish.sh', 'example.invalid/artifact')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([c['args'][:2] for c in calls[:2]],
                         [['image', 'inspect'], ['image', 'inspect']])
        pushed = [c['args'][1] for c in calls if c['args'][0] == 'push']
        self.assertEqual(pushed, [f'example.invalid/artifact:{target}{suffix}'
                                 for target in ('artifact', 'comparisons')
                                 for suffix in ('-' + self.revision[:12], '')])

    def test_missing_or_stale_second_image_prevents_all_pushes(self):
        for kind in ('STALE_IMAGE', 'MISSING_IMAGE'):
            with self.subTest(kind=kind):
                self.log.unlink(missing_ok=True)
                result, calls = self.invoke('publish.sh', **{kind: 'looplibrarian:comparisons'})
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c['args'][0] in ('tag', 'push') for c in calls))

    def test_failed_upload_stops_before_updating_stable_tags(self):
        result, calls = self.invoke('publish.sh', FAIL_PUSH='1')
        self.assertNotEqual(result.returncode, 0)
        pushed = [c for c in calls if c['args'][0] == 'push']
        self.assertEqual(len(pushed), 1)
        self.assertTrue(pushed[0]['args'][1].endswith('artifact-' + self.revision[:12]))


if __name__ == '__main__':
    unittest.main()
