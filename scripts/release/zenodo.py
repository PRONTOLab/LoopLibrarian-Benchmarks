#!/usr/bin/env python3
"""Upload a verified source archive as a new version of an existing Zenodo record."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

from package import digest, TAG_PATTERN

MARKER = 'LoopLibrarian automated source release: '
METADATA_FIELDS = {
    'upload_type', 'publication_type', 'image_type', 'publication_date', 'title',
    'creators', 'description', 'access_right', 'license', 'embargo_date',
    'access_conditions', 'keywords', 'notes', 'related_identifiers', 'contributors',
    'references', 'communities', 'grants', 'journal_title', 'journal_volume',
    'journal_issue', 'journal_pages', 'conference_title', 'conference_acronym',
    'conference_dates', 'conference_place', 'conference_url', 'conference_session',
    'conference_session_part', 'imprint_publisher', 'imprint_isbn', 'imprint_place',
    'partof_title', 'partof_pages', 'thesis_supervisors', 'thesis_university',
    'subjects', 'version', 'language', 'locations', 'dates', 'method',
}


class ApiError(RuntimeError):
    def __init__(self, method, path, status, response_format):
        self.status = status
        super().__init__(f'Zenodo {method} {path} returned HTTP {status} ({response_format} response)')


class Client:
    def __init__(self, base, token):
        if base not in ('https://zenodo.org', 'https://sandbox.zenodo.org'):
            raise ValueError('Expected the production or sandbox Zenodo HTTPS endpoint')
        if not token:
            raise ValueError('Set ZENODO_TOKEN in the environment')
        self.base, self.token = base, token

    def request(self, method, path, data=None, file=None):
        url = urllib.parse.urljoin(self.base, path)
        parsed = urllib.parse.urlparse(url)
        if (parsed.scheme != 'https' or parsed.netloc != urllib.parse.urlparse(self.base).netloc
                or not parsed.path.startswith('/api/') or parsed.username or parsed.password):
            raise ValueError('Refusing to send credentials outside the configured Zenodo API')
        headers = {'Authorization': 'Bearer ' + self.token,
                   'User-Agent': 'LoopLibrarian-release/1.0 (https://github.com/PRONTOLab/LoopLibrarian-Benchmarks)'}
        body = None
        stream = None
        if data is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(data).encode()
        if file is not None:
            headers.update({'Content-Type': 'application/octet-stream',
                            'Content-Length': str(Path(file).stat().st_size)})
            stream = Path(file).open('rb')
            body = stream
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        # Do not follow a redirect with a bearer token to another origin.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                return None
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=300) as response:
                payload = response.read()
                return json.loads(payload) if payload else None
        except urllib.error.HTTPError as exc:
            # Server response bodies and request headers are never printed.
            kind = 'JSON' if 'application/json' in exc.headers.get('Content-Type', '') else 'non-JSON'
            raise ApiError(method, parsed.path, exc.code, kind) from None
        except urllib.error.URLError:
            raise RuntimeError(f'Zenodo {method} {parsed.path} failed; rerun to resume') from None
        finally:
            if stream:
                stream.close()


def version(row):
    return str(row.get('metadata', {}).get('version', '')).removeprefix('v')


def published(row):
    return bool(row.get('submitted')) or row.get('state') == 'done'


def file_name(row):
    return row.get('filename', row.get('key'))


def files_match(row, manifest):
    files = row.get('files', [])
    return (len(files) == 1 and file_name(files[0]) == manifest['archive'] and
            str(files[0].get('checksum', '')).removeprefix('md5:') == manifest['md5'] and
            int(files[0].get('filesize', files[0].get('size', -1))) == manifest['size'])


def marker(manifest):
    return MARKER + json.dumps({key: manifest[key] for key in
                               ('tag', 'benchmark_commit', 'compiler_commit', 'sha256')}, sort_keys=True)


def metadata_for(latest, manifest):
    metadata = {key: value for key, value in latest['metadata'].items() if key in METADATA_FIELDS}
    metadata.update(upload_type='software', version=manifest['tag'].removeprefix('v'),
                    publication_date=date.today().isoformat())
    notes = str(metadata.get('notes', '')).split(MARKER, 1)[0].rstrip()
    metadata['notes'] = (notes + '\n\n' if notes else '') + marker(manifest)
    url = manifest['repository'] + '/tree/' + manifest['tag']
    related = [row for row in metadata.get('related_identifiers', [])
               if not (row.get('relation') == 'isIdenticalTo' and
                       row.get('identifier', '').startswith(manifest['repository']))]
    related.append(dict(identifier=url, relation='isIdenticalTo', scheme='url'))
    metadata['related_identifiers'] = related
    return metadata


def family(client, base_id):
    try:
        base = client.request('GET', f'/api/deposit/depositions/{base_id}')
    except ApiError as exc:
        if exc.status != 403:
            raise
        # Distinguish access to this record from rejection of all deposit access.
        # Read only; never print the user's other deposition metadata.
        try:
            client.request('GET', '/api/deposit/depositions?size=1')
        except ApiError as account_error:
            raise RuntimeError(f'{exc}; account-list check also failed: {account_error}. '
                               'Check token scopes and whether Zenodo is blocking this runner.') from None
        raise RuntimeError(f'The token can list depositions but cannot manage record {base_id}. '
                           'Use a token from an account with access to this record.') from None
    concept = str(base.get('conceptrecid', ''))
    if not concept:
        raise ValueError('Zenodo did not identify the existing record family')
    rows = {str(base['id']): base}
    # List only the user's accessible depositions; never create a new DOI family.
    for page in range(1, 101):
        batch = client.request('GET', f'/api/deposit/depositions?size=100&all_versions=1&page={page}')
        if not isinstance(batch, list):
            raise ValueError('Unexpected Zenodo deposition-list response')
        for row in batch:
            if str(row.get('conceptrecid')) == concept:
                rows[str(row['id'])] = row
        if len(batch) < 100:
            return list(rows.values())
    raise ValueError('Too many deposition pages; narrow the configured account')


def publish_archive(client, base_id, manifest, archive, publish=False):
    archive = Path(archive)
    if not re.fullmatch(TAG_PATTERN, manifest['tag']):
        raise ValueError('Invalid release version')
    if (archive.name != manifest['archive'] or archive.stat().st_size != manifest['size']
            or digest(archive) != manifest['sha256'] or digest(archive, 'md5') != manifest['md5']):
        raise ValueError('Release archive differs from its checked manifest')
    rows = family(client, base_id)
    wanted = manifest['tag'].removeprefix('v')
    for row in rows:
        if published(row) and version(row) == wanted:
            if marker(manifest) not in row['metadata'].get('notes', '') or not files_match(row, manifest):
                raise ValueError('This version is already published with different content; use a new tag')
            return dict(status='already-published', deposition_id=row['id'],
                        doi=row.get('doi'), url=row.get('doi_url') or row['links'].get('html'))
    previous = [row for row in rows if published(row)]
    if not previous:
        raise ValueError('The configured Zenodo record has no published version')
    latest = max(previous, key=lambda row: int(row['id']))
    drafts = [row for row in rows if not published(row)]
    if len(drafts) > 1:
        raise ValueError('Multiple unpublished drafts exist; resolve them in Zenodo first')
    if drafts:
        draft = drafts[0]
        if marker(manifest) not in draft['metadata'].get('notes', '') or version(draft) != wanted:
            raise ValueError('An unrelated Zenodo draft exists; it has been left untouched')
    else:
        response = client.request('POST', f"/api/deposit/depositions/{latest['id']}/actions/newversion")
        draft = client.request('GET', response['links']['latest_draft'])
        draft = client.request('PUT', f"/api/deposit/depositions/{draft['id']}",
                               data={'metadata': metadata_for(latest, manifest)})
    draft_id = draft['id']
    if not files_match(draft, manifest):
        # A new version inherits its predecessor's files. Only this draft is modified.
        for file in draft.get('files', []):
            client.request('DELETE', f"/api/deposit/depositions/{draft_id}/files/{file['id']}")
        response = client.request('PUT', draft['links']['bucket'] + '/' +
                                  urllib.parse.quote(archive.name, safe=''), file=archive)
        if (str(response.get('checksum', '')).removeprefix('md5:') != manifest['md5'] or
                int(response.get('size', -1)) != manifest['size']):
            raise ValueError('Zenodo upload checksum/size mismatch; draft was not published')
    draft = client.request('GET', f'/api/deposit/depositions/{draft_id}')
    if not files_match(draft, manifest) or marker(manifest) not in draft['metadata'].get('notes', ''):
        raise ValueError('Zenodo draft verification failed; draft was not published')
    if publish:
        draft = client.request('POST', f'/api/deposit/depositions/{draft_id}/actions/publish')
        if not published(draft):
            raise RuntimeError('Zenodo accepted publication but has not confirmed it; rerun to check')
    return dict(status='published' if publish else 'draft', deposition_id=draft_id,
                doi=draft.get('doi'), url=draft.get('doi_url') or draft['links'].get('html'),
                tag=manifest['tag'], archive_sha256=manifest['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-id', required=True, type=int)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--endpoint', choices=['https://zenodo.org', 'https://sandbox.zenodo.org'],
                        default='https://zenodo.org')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    try:
        client = Client(args.endpoint, os.environ.get('ZENODO_TOKEN'))
        manifest = json.loads(args.manifest.read_text())
        result = publish_archive(client, args.record_id, manifest,
                                 args.manifest.parent / manifest['archive'], args.publish)
    except (ValueError, RuntimeError) as exc:
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            message = str(exc).replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
            print('::error::' + message, flush=True)
        parser.exit(1, str(exc) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f"status={result['status']}\ndeposition_id={int(result['deposition_id'])}\n")
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(f"Zenodo: {result['status']} — {result.get('url') or result['deposition_id']}\n")


if __name__ == '__main__':
    main()
