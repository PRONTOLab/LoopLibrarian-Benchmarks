#!/usr/bin/env python3
"""Download the public CUDA headers used by the recorded source imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile


def payloads(archive, package, work):
    if package.get('format') != 'conda':
        return [archive]
    result = []
    with zipfile.ZipFile(archive) as zipped:
        for index, name in enumerate(zipped.namelist()):
            if not name.endswith('.tar.zst'):
                continue
            compressed = Path(work) / f'{index}.tar.zst'
            compressed.write_bytes(zipped.read(name))
            output = compressed.with_suffix('')
            subprocess.run(['zstd', '-q', '-d', str(compressed), '-o', str(output)], check=True)
            result.append(output)
    if not result:
        raise ValueError('No payload in CUDA header package')
    return result


def prepare(destination, manifest=Path(__file__).with_name('cuda-headers.json')):
    destination = Path(destination).resolve()
    record = json.loads(Path(manifest).read_text())
    for package in record['packages']:
        target = destination / package['directory']
        if not target.is_relative_to(destination):
            raise ValueError('Header destination escapes the dependency directory')
        with tempfile.TemporaryDirectory() as work:
            archive = Path(work) / 'package.tar.xz'
            urllib.request.urlretrieve(package['url'], archive)
            digest = hashlib.sha256()
            with archive.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != package['sha256']:
                raise RuntimeError('CUDA header archive checksum mismatch')
            files = []
            for payload in payloads(archive, package, work):
                with tarfile.open(payload) as tar:
                    for member in tar.getmembers():
                        path = PurePosixPath(member.name)
                        if path.is_absolute() or '..' in path.parts:
                            raise ValueError('Unsafe CUDA archive path')
                        if 'include' in path.parts:
                            relative = Path(*path.parts[path.parts.index('include'):])
                        elif 'license' in path.name.lower() or 'eula' in path.name.lower():
                            relative = Path(path.name)
                        else:
                            continue
                        if member.isdir():
                            continue
                        if not member.isfile():
                            raise ValueError('Unexpected link in CUDA header archive')
                        output = target / relative
                        if not output.resolve().is_relative_to(destination):
                            raise ValueError('Header destination escapes the dependency directory')
                        data = tar.extractfile(member).read()
                        if output.exists() and output.read_bytes() != data:
                            raise RuntimeError(f'{output}: existing header differs; left untouched')
                        files.append((output, data))
            # Check existing files before writing any files from this package.
            for output, data in files:
                output.parent.mkdir(parents=True, exist_ok=True)
                if not output.exists():
                    output.write_bytes(data)
            print(f"Prepared {package['component']}: {len(files)} headers/license files", flush=True)


if __name__ == '__main__':
    default = Path(os.environ.get('POLYGEIST_DEPENDENCY_ROOT',
                                  Path(__file__).resolve().parents[1] / 'third_party')) / 'cuda_headers'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, nargs='?', default=default)
    args = parser.parse_args()
    prepare(args.destination)
