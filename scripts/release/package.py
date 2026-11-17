#!/usr/bin/env python3
"""Package a tagged benchmark revision and its exact Polygeist source revision."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import tempfile
import zipfile
import zlib

COMPILER_URL = 'https://github.com/arpitj1/Polygeist.git'
REPOSITORY_URL = 'https://github.com/PRONTOLab/LoopLibrarian-Benchmarks'
TAG_PATTERN = r'v[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?'


def git(repo, *args, input=None):
    return subprocess.check_output(['git', '-C', str(repo), *args],
                                   input=input, stderr=subprocess.PIPE)


def tree(repo, commit):
    rows = []
    for entry in git(repo, 'ls-tree', '-rz', '-t', commit).split(b'\0'):
        if entry:
            header, name = entry.split(b'\t', 1)
            mode, kind, oid = header.decode().split()
            rows.append((mode, kind, oid, name.decode()))
    return rows


def export(repo, commit, destination):
    """Copy committed files and reconstruct a credential-free shallow Git snapshot.

    Existing setup/build/provenance scripts need real commit IDs and Git trees.
    Only this commit and its tree/blob objects are copied, never local config,
    hooks, logs, credentials, untracked files, or parent history.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(git(repo, 'archive', '--format=tar', commit))) as archive:
        for item in archive:
            name = PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts or '.git' in name.parts:
                raise ValueError('Unsafe tracked archive path')
            if item.issym():
                link = PurePosixPath(item.linkname)
                if link.is_absolute() or '..' in link.parts:
                    raise ValueError('Archive contains an external symlink')
            elif not (item.isfile() or item.isdir()):
                raise ValueError('Unsupported archive entry')
            archive.extract(item, destination, filter='data')

    entries = tree(repo, commit)
    objects = sorted({commit, *(oid for _, kind, oid, _ in entries if kind != 'commit')})
    metadata = destination / '.git'
    (metadata / 'objects').mkdir(parents=True)
    (metadata / 'refs').mkdir()
    (metadata / 'HEAD').write_text(commit + '\n')
    (metadata / 'shallow').write_text(commit + '\n')
    (metadata / 'config').write_text(
        '[core]\n\trepositoryformatversion = 0\n\tbare = false\n\tfilemode = true\n')
    # ls-tree omits the root tree itself.
    objects.append(git(repo, 'rev-parse', commit + '^{tree}').decode().strip())
    with subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        for oid in sorted(set(objects)):
            proc.stdin.write((oid + '\n').encode())
            proc.stdin.flush()
            actual, kind, size = proc.stdout.readline().decode().split()
            data = proc.stdout.read(int(size))
            if proc.stdout.read(1) != b'\n' or actual != oid or len(data) != int(size):
                raise ValueError('Incomplete Git object')
            raw = kind.encode() + b' ' + size.encode() + b'\0' + data
            if hashlib.sha1(raw).hexdigest() != oid:
                raise ValueError('Git object checksum mismatch')
            output = metadata / 'objects' / oid[:2] / oid[2:]
            output.parent.mkdir(exist_ok=True)
            output.write_bytes(zlib.compress(raw))
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError('Cannot export Git objects')
    # Build the index from object IDs without reading host-specific file stats.
    git(destination, '-c', 'index.version=2', 'update-index', '-z', '--index-info',
        input=git(repo, 'ls-tree', '-rz', commit))
    # Diff needs refreshed stats to check contents. Preserve the portable index
    # rather than archiving the host-specific inode/timestamp data it writes.
    index = metadata / 'index'
    portable_index = index.read_bytes()
    git(destination, 'diff', '--exit-code', 'HEAD', '--')
    index.write_bytes(portable_index)
    return [{'path': name, 'revision': oid} for _, kind, oid, name in entries if kind == 'commit']


def write_zip(source, output):
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(source.rglob('*')):
            directory = path.is_dir() and not path.is_symlink()
            name = source.name + '/' + str(path.relative_to(source))
            info = zipfile.ZipInfo(name + ('/' if directory else ''),
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = path.lstat().st_mode
            if directory:
                info.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10
                data = b''
            elif path.is_symlink():
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                data = os.readlink(path).encode()
            else:
                info.external_attr = (stat.S_IFREG | (0o755 if mode & 0o111 else 0o644)) << 16
                data = path.read_bytes()
            archive.writestr(info, data)


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def package(repository, tag, output, compiler_repository=None):
    if not re.fullmatch(TAG_PATTERN, tag):
        raise ValueError('Expected a version tag such as v0.4.0')
    repository, output = Path(repository).resolve(), Path(output).resolve()
    commit = git(repository, 'rev-parse', 'refs/tags/' + tag + '^{commit}').decode().strip()
    compiler = git(repository, 'rev-parse', commit + ':third_party/Polygeist').decode().strip()
    modules = [row for row in tree(repository, commit) if row[1] == 'commit']
    if [(row[2], row[3]) for row in modules] != [(compiler, 'third_party/Polygeist')]:
        raise ValueError('Unexpected benchmark submodules; update the packaging policy')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='looplibrarian-release-') as tmp:
        tmp = Path(tmp)
        if compiler_repository is None:
            compiler_repository = tmp / 'compiler.git'
            subprocess.run(['git', 'init', '--bare', '--quiet', str(compiler_repository)], check=True)
            subprocess.run(['git', '-C', str(compiler_repository), 'fetch', '--quiet',
                            '--depth=1', COMPILER_URL, compiler], check=True)
        else:
            compiler_repository = Path(compiler_repository).resolve()
        git(compiler_repository, 'cat-file', '-e', compiler + '^{commit}')
        source = tmp / ('LoopLibrarian-Benchmarks-' + tag)
        export(repository, commit, source)
        nested = export(compiler_repository, compiler, source / 'third_party/Polygeist')
        for name in ('README.md', 'third_party/Polygeist/CMakeLists.txt',
                     'third_party/Polygeist/lib/polygeist/Passes/RaiseToLinalg.cpp'):
            if not (source / name).is_file():
                raise ValueError('Missing release source: ' + name)
        manifest = dict(format=1, tag=tag, benchmark_commit=commit,
                        compiler_commit=compiler, repository=REPOSITORY_URL,
                        compiler_repository=COMPILER_URL,
                        external_submodules=nested,
                        git_metadata='shallow snapshots reconstructed from committed objects')
        (source / 'RELEASE-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (source / 'ARCHIVE-README.txt').write_text(
            'This archive includes the benchmark release and its pinned Polygeist sources.\n'
            'Extract with unzip so executable permissions and any symlinks are preserved.\n'
            'Clean shallow Git snapshots retain original commit IDs for the existing\n'
            'setup, Docker build, and provenance commands. No local Git configuration,\n'
            'hooks, credentials, build products, or parent history are included.\n'
            'Follow README.md. Polygeist is already populated; skip its submodule\n'
            'initialization command. Public LLVM and other dependencies still use the\n'
            'documented setup/download steps. RELEASE-MANIFEST.json records revisions.\n')
        archive = output / (source.name + '.zip')
        write_zip(source, archive)
        with zipfile.ZipFile(archive) as check:
            if check.testzip() is not None:
                raise ValueError('Release ZIP failed its integrity check')
        manifest.update(archive=archive.name, size=archive.stat().st_size,
                        sha256=digest(archive), md5=digest(archive, 'md5'))
        report = output / (source.name + '.json')
        report.write_text(json.dumps(manifest, indent=2) + '\n')
        archive.with_suffix('.zip.sha256').write_text(manifest['sha256'] + '  ' + archive.name + '\n')
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler-repository', type=Path)
    args = parser.parse_args()
    result = package(args.repository, args.tag, args.output, args.compiler_repository)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
