#!/usr/bin/env python3
"""Fetch public LLVM sources and apply the artifact's recorded parser patch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def git(source, *args, env=None, check=True):
    result = subprocess.run(['git', '-C', str(source), *map(str, args)],
                            text=True, capture_output=True, env=env)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def prepare(source, manifest=Path(__file__).with_name('llvm.json')):
    source = Path(source).resolve()
    manifest = Path(manifest).resolve()
    record = json.loads(manifest.read_text())
    patch = manifest.parent / record['patch']
    if hashlib.sha256(patch.read_bytes()).hexdigest() != record['patch_sha256']:
        raise RuntimeError('LLVM patch checksum mismatch')
    source.mkdir(parents=True, exist_ok=True)
    if not (source / '.git').exists():
        if any(source.iterdir()):
            raise RuntimeError(f'{source}: not an empty directory or Git checkout')
        git(source, 'init')
    head = git(source, 'rev-parse', '--verify', 'HEAD', check=False)
    if head.returncode:
        git(source, 'fetch', '--depth', '1', record['url'], record['revision'])
        git(source, 'checkout', '--detach', 'FETCH_HEAD')
    elif head.stdout.strip() not in (record['revision'], record['patched_revision']):
        raise RuntimeError(f'{source}: unexpected LLVM revision; checkout left untouched')

    # Construct the expected tree in a temporary index without changing the
    # user's index, and verify it matches the recorded source tree exactly.
    with tempfile.TemporaryDirectory() as work:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(work) / 'index'))
        git(source, 'read-tree', record['revision'], env=env)
        git(source, 'apply', '--cached', patch, env=env)
        tree = git(source, 'write-tree', env=env).stdout.strip()
    if tree != record['tree']:
        raise RuntimeError('LLVM patch does not produce the recorded source tree')
    if git(source, 'diff', '--quiet', tree, '--', check=False).returncode:
        if git(source, 'diff', '--quiet', record['revision'], '--', check=False).returncode:
            raise RuntimeError(f'{source}: local LLVM changes differ; checkout left untouched')
        git(source, 'apply', '--check', patch)
        git(source, 'apply', patch)
    git(source, 'diff', '--exit-code', tree, '--')
    print(f'LLVM source tree verified: {tree}', flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    prepare(args.source)
