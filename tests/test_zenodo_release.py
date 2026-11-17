"""Release archives use pinned sources; publication never overwrites other work."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.parse
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/release'))
import package as release_package
import zenodo


def git(repo, *args):
    return subprocess.check_output(
        ['git', '-C', str(repo), '-c', 'user.name=Release Test',
         '-c', 'user.email=release@example.invalid', '-c', 'commit.gpgsign=false',
         '-c', 'core.hooksPath=/dev/null', *args], stderr=subprocess.PIPE).decode().strip()


class SourceArchive(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.compiler = self.root / 'compiler'
        self.repository = self.root / 'benchmark'
        for repo in (self.compiler, self.repository):
            repo.mkdir()
            git(repo, 'init', '-q')
        for name, text in [('CMakeLists.txt', 'pinned compiler\n'),
                           ('lib/polygeist/Passes/RaiseToLinalg.cpp', '// source\n')]:
            file = self.compiler / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)
        git(self.compiler, 'add', '.')
        git(self.compiler, 'update-index', '--add', '--cacheinfo',
            '160000,' + '1' * 40 + ',llvm-project')
        git(self.compiler, 'commit', '-qm', 'Pinned compiler')
        self.compiler_commit = git(self.compiler, 'rev-parse', 'HEAD')
        (self.compiler / 'CMakeLists.txt').write_text('newer compiler\n')
        git(self.compiler, 'commit', '-qam', 'Not pinned')
        (self.repository / 'README.md').write_text('tagged benchmark\n')
        (self.repository / 'run.sh').write_text('#!/bin/sh\nexit 0\n')
        (self.repository / 'run.sh').chmod(0o755)
        (self.repository / 'readme-link').symlink_to('README.md')
        git(self.repository, 'add', '.')
        git(self.repository, 'update-index', '--add', '--cacheinfo',
            '160000,' + self.compiler_commit + ',third_party/Polygeist')
        git(self.repository, 'commit', '-qm', 'Release')
        self.commit = git(self.repository, 'rev-parse', 'HEAD')
        git(self.repository, 'tag', 'v0.4.0')
        (self.repository / 'README.md').write_text('uncommitted change\n')
        (self.repository / 'local-secret').write_text('do not archive\n')
        git(self.repository, 'config', 'http.extraheader', 'Authorization: private-test-value')

    def package(self, destination):
        return release_package.package(self.repository, 'v0.4.0', destination, self.compiler)

    def test_pinned_sources_and_clean_git_metadata_survive_unzip(self):
        output = self.root / 'dist'
        manifest = self.package(output)
        archive = output / manifest['archive']
        extracted = self.root / 'extracted'
        subprocess.run(['unzip', '-q', str(archive), '-d', str(extracted)], check=True)
        source = extracted / 'LoopLibrarian-Benchmarks-v0.4.0'
        self.assertEqual((source / 'README.md').read_text(), 'tagged benchmark\n')
        self.assertEqual((source / 'third_party/Polygeist/CMakeLists.txt').read_text(),
                         'pinned compiler\n')
        self.assertFalse((source / 'local-secret').exists())
        self.assertNotIn('private-test-value', (source / '.git/config').read_text())
        self.assertEqual(git(source, 'rev-parse', 'HEAD'), self.commit)
        self.assertEqual(git(source / 'third_party/Polygeist', 'rev-parse', 'HEAD'), self.compiler_commit)
        git(source, 'diff', '--exit-code', 'HEAD')
        git(source / 'third_party/Polygeist', 'diff', '--exit-code', 'HEAD')
        self.assertTrue((source / 'run.sh').stat().st_mode & 0o111)
        self.assertTrue((source / 'readme-link').is_symlink())
        self.assertEqual(manifest['external_submodules'],
                         [{'path': 'llvm-project', 'revision': '1' * 40}])
        self.assertEqual(manifest['sha256'], release_package.digest(archive))
        self.assertEqual(json.loads((source / 'RELEASE-MANIFEST.json').read_text())['compiler_commit'],
                         self.compiler_commit)
        with zipfile.ZipFile(archive) as zipped:
            self.assertFalse(any('/.git/hooks/' in name for name in zipped.namelist()))

    def test_repeated_packages_have_identical_checksums(self):
        first = self.package(self.root / 'first')
        second = self.package(self.root / 'second')
        self.assertEqual(first, second)

    def test_rejects_branch_and_shell_syntax_as_tags(self):
        for tag in ('main', '../v0.4.0', 'v0.4.0$(id)', 'v0.4.0;echo bad'):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release_package.package(self.repository, tag, self.root / 'dist', self.compiler)


class FakeZenodo:
    def __init__(self, manifest):
        self.manifest = manifest
        self.calls = []
        self.bad_checksum = False
        self.rows = {10: dict(id=10, conceptrecid='9', submitted=True, state='done',
                             metadata=dict(version='0.3.0', title='Artifact',
                                           creators=[{'name': 'Original Author'}],
                                           description='Original description', license='MIT'),
                             files=[dict(id='old', filename='old.zip', checksum='old', filesize=3)],
                             links=dict(html='https://zenodo.org/records/10'))}

    def request(self, method, path, data=None, file=None):
        self.calls.append((method, path))
        parsed = urllib.parse.urlparse(path)
        parts = parsed.path.strip('/').split('/')
        if parsed.path == '/api/deposit/depositions':
            assert urllib.parse.parse_qs(parsed.query)['all_versions'] == ['1']
            return copy.deepcopy(list(self.rows.values()))
        if parts[:2] == ['api', 'files']:
            assert method == 'PUT'
            checksum = 'incorrect' if self.bad_checksum else self.manifest['md5']
            row = dict(id='uploaded', filename=Path(file).name, checksum='md5:' + checksum,
                       filesize=Path(file).stat().st_size)
            self.rows[11]['files'].append(row)
            return dict(checksum=row['checksum'], size=row['filesize'])
        record_id = int(parts[3])
        if method == 'GET':
            return copy.deepcopy(self.rows[record_id])
        if method == 'POST' and parts[-1] == 'newversion':
            assert record_id == 10
            self.rows[11] = copy.deepcopy(self.rows[10])
            self.rows[11].update(id=11, submitted=False, state='unsubmitted',
                                 links=dict(bucket='https://zenodo.org/api/files/bucket',
                                            html='https://zenodo.org/deposit/11'))
            # Zenodo returns the old resource with a link to the new draft.
            response = copy.deepcopy(self.rows[10])
            response['links']['latest_draft'] = 'https://zenodo.org/api/deposit/depositions/11'
            return response
        if method == 'PUT':
            self.rows[record_id]['metadata'] = copy.deepcopy(data['metadata'])
        elif method == 'DELETE':
            assert record_id == 11, 'Never delete published files'
            self.rows[record_id]['files'] = [row for row in self.rows[record_id]['files']
                                             if row['id'] != parts[-1]]
            return None
        elif method == 'POST' and parts[-1] == 'publish':
            self.rows[record_id].update(submitted=True, state='done', doi='10.example/test')
        else:
            raise AssertionError((method, path))
        return copy.deepcopy(self.rows[record_id])


class ZenodoPublication(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.archive = Path(temporary.name) / 'LoopLibrarian-Benchmarks-v0.4.0.zip'
        self.archive.write_bytes(b'verified archive fixture')
        self.manifest = dict(tag='v0.4.0', archive=self.archive.name,
                             size=self.archive.stat().st_size,
                             sha256=release_package.digest(self.archive),
                             md5=release_package.digest(self.archive, 'md5'),
                             benchmark_commit='a' * 40, compiler_commit='b' * 40,
                             repository=release_package.REPOSITORY_URL)
        self.client = FakeZenodo(self.manifest)

    def run_release(self, publish=False):
        return zenodo.publish_archive(self.client, 10, self.manifest, self.archive, publish)

    def test_draft_default_then_publish_and_repeat_without_duplicate(self):
        old = copy.deepcopy(self.client.rows[10])
        result = self.run_release()
        self.assertEqual(result['status'], 'draft')
        self.assertEqual(self.client.rows[10], old)
        self.assertTrue(zenodo.files_match(self.client.rows[11], self.manifest))
        self.assertEqual(self.client.rows[11]['metadata']['creators'], old['metadata']['creators'])
        self.assertEqual(self.client.rows[11]['metadata']['version'], '0.4.0')
        self.assertFalse(any(path.endswith('/publish') for _, path in self.client.calls))
        self.assertEqual(self.run_release(True)['status'], 'published')
        self.client.calls.clear()
        self.assertEqual(self.run_release(True)['status'], 'already-published')
        self.assertTrue(all(method == 'GET' for method, _ in self.client.calls))

    def test_checksum_failure_never_publishes_and_retry_resumes(self):
        self.client.bad_checksum = True
        with self.assertRaisesRegex(ValueError, 'checksum/size mismatch'):
            self.run_release(True)
        self.assertFalse(any(path.endswith('/publish') for _, path in self.client.calls))
        self.client.bad_checksum = False
        self.client.calls.clear()
        self.assertEqual(self.run_release(True)['status'], 'published')
        self.assertFalse(any(path.endswith('/newversion') for _, path in self.client.calls))

    def test_unrelated_draft_is_untouched(self):
        self.client.rows[11] = copy.deepcopy(self.client.rows[10])
        self.client.rows[11].update(id=11, submitted=False, state='unsubmitted')
        original = copy.deepcopy(self.client.rows)
        with self.assertRaisesRegex(ValueError, 'unrelated Zenodo draft'):
            self.run_release(True)
        self.assertEqual(self.client.rows, original)
        self.assertTrue(all(method == 'GET' for method, _ in self.client.calls))

    def test_conflicting_published_version_is_untouched(self):
        self.client.rows[10]['metadata']['version'] = '0.4.0'
        with self.assertRaisesRegex(ValueError, 'already published with different content'):
            self.run_release(True)
        self.assertTrue(all(method == 'GET' for method, _ in self.client.calls))

    def test_changed_local_archive_makes_no_api_calls(self):
        self.archive.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs from its checked manifest'):
            self.run_release(True)
        self.assertEqual(self.client.calls, [])

    def test_record_forbidden_is_distinguished_from_account_access(self):
        record_error = zenodo.ApiError('GET', '/api/deposit/depositions/10', 403, 'JSON')
        with patch.object(self.client, 'request', side_effect=[record_error, []]) as request:
            with self.assertRaisesRegex(RuntimeError, 'can list depositions but cannot manage record 10'):
                self.run_release()
            self.assertTrue(all(call.args[0] == 'GET' for call in request.call_args_list))

    def test_general_api_rejection_reports_both_read_only_checks(self):
        error = zenodo.ApiError('GET', '/api/deposit/depositions', 403, 'non-JSON')
        with patch.object(self.client, 'request', side_effect=[error, error]) as request:
            with self.assertRaisesRegex(RuntimeError, 'account-list check also failed'):
                self.run_release()
            self.assertEqual(request.call_count, 2)
            self.assertTrue(all(call.args[0] == 'GET' for call in request.call_args_list))

    def test_token_is_never_sent_to_another_origin(self):
        client = zenodo.Client('https://zenodo.org', 'test-token')
        for url in ('https://example.com/api/files/x', '//example.com/api/files/x',
                    'http://zenodo.org/api/files/x', 'https://zenodo.org/account/settings',
                    'https://user:password@zenodo.org/api/files/x'):
            with self.subTest(url=url), patch('urllib.request.build_opener') as opener:
                with self.assertRaises(ValueError):
                    client.request('PUT', url, file=self.archive)
                opener.assert_not_called()


if __name__ == '__main__':
    unittest.main()
