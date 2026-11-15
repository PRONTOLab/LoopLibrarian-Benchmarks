#!/usr/bin/env python3
"""Package built GPU experiments and execute them on a local or SSH target."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent
SUITES = ('polybench_gpu', 'llama', 'mfem', 'aten')
DEFAULT_LIBRARIES = '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def build_files(suite, source):
    """Select execution inputs, keeping every proof and build record unchanged."""
    files = ['provenance.json']
    executables = []
    if suite == 'polybench_gpu':
        files += ['cases.json', 'paper_targets.csv', 'final-provenance.json']
        for case in read(source/'cases.json'):
            build = read(source/case/'build.json')
            if build['source_build_rc'] or build['gpu_build_rc']:
                raise ValueError('Unsuccessful build: ' + case)
            baseline = 'paper-baseline' if case in ('correlation', 'covariance') else 'native'
            files += [f'{case}/{name}' for name in
                      ('build.json', 'export/match.json', 'gpu/gpu_resident.mlir',
                       baseline+'-build.json')]
            executables += [f'{case}/{name}' for name in ('reference', baseline, 'raised')]
            if sha(source/case/'raised') != build['raised_sha256']:
                raise ValueError('Executable differs from build: ' + case)
    elif suite == 'llama':
        build = read(source/'build.json')
        files += ['build.json', 'coverage.json']
        executables += list(build['binaries'])
        for name, expected in build['binaries'].items():
            if sha(source/name) != expected:
                raise ValueError('Executable differs from build: ' + name)
    elif suite == 'mfem':
        for path in sorted(source.glob('*/build.json')):
            case = path.parent.name
            if sha(path.parent/'program') != read(path)['executable_sha256']:
                raise ValueError('Executable differs from build: ' + case)
            files.append(f'{case}/build.json')
            executables.append(f'{case}/program')
        if not executables:
            raise ValueError('No MFEM builds found')
    else:
        selection = read(source/'selection.json')
        if selection['cohort'] != 'paper':
            raise ValueError('ATen execution bundles currently support the paper selection')
        files += ['selection.json', 'matching/provenance.json']
        for case in selection['cases']:
            files.append(f'matching/{case}/result.json')
            files += [str(p.relative_to(source)) for p in
                      sorted((source/'matching'/case).glob('rank*/proof.json'))]
            path = source/'gpu'/case/'build.json'
            if not path.exists():
                continue  # Unmatched cases remain in the report.
            build = read(path)
            files.append(f'gpu/{case}/build.json')
            if not build['rc']:
                program = f'gpu/{case}/raised-program'
                if sha(source/program) != build['binary_sha256']:
                    raise ValueError('Executable differs from build: ' + case)
                executables.append(program)
            files += [str(p.relative_to(source)) for p in
                      [path.parent/'work/match.json', path.parent/'work/abi.json'] if p.exists()]
    return sorted(set(files + executables)), executables


def pack(destination, builds, ggml_build=None, root=ROOT):
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Use a new bundle directory: ' + str(destination))
    destination.mkdir(parents=True)
    manifest = dict(format=1, suites={}, files={}, executables=[], runtime_sources={})
    try:
        # Only first-party tracked source files; never SDKs, credentials, or saved runs.
        tracked = subprocess.check_output(['git', '-c', 'safe.directory='+str(root.resolve()),
                                           'ls-files', '-z'], cwd=root).decode().split('\0')
        allowed = {'scripts', 'polybench', 'llama', 'mfem', 'aten'}
        sources = sorted({name for name in tracked if name and
                          (Path(name).parts[0] in allowed or name in ('reproduce.py', 'README.md', 'LICENSE'))}
                         | {'target.py'})
        for name in sources:
            path = root/name
            if not path.is_file() or path.is_symlink():
                raise ValueError('Expected a regular source file: ' + name)
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            manifest['runtime_sources'][name] = sha(target)
        for suite, source in builds.items():
            files, executables = build_files(suite, source)
            for name in files:
                target = destination/'builds'/suite/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, target)
            manifest['suites'][suite] = {'build_provenance': read(source/'provenance.json')}
            manifest['executables'] += [f'builds/{suite}/{name}' for name in executables]
        if 'llama' in builds:
            if ggml_build is None:
                raise ValueError('Llama needs --ggml-build containing the built AArch64 ggml libraries')
            libraries = {p.name: p for p in ggml_build.rglob('libggml*.so.0')}
            if not {'libggml.so.0', 'libggml-base.so.0', 'libggml-cpu.so.0', 'libggml-cuda.so.0'} <= libraries.keys():
                raise ValueError('Missing ggml shared libraries')
            for name, path in libraries.items():
                target = destination/'runtime/ggml/ggml/src'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path.resolve(), target)
        manifest['files'] = {str(p.relative_to(destination)): sha(p)
                             for p in sorted(destination.rglob('*')) if p.is_file()}
        save(destination/'execution-bundle.json', manifest)
    except Exception:
        shutil.rmtree(destination)
        raise
    return manifest


def verify(bundle):
    manifest = read(bundle/'execution-bundle.json')
    if manifest.get('format') != 1:
        raise ValueError('Unsupported execution bundle format')
    for name, expected in manifest['files'].items():
        path = bundle/name
        if not path.resolve().is_relative_to(bundle.resolve()) or path.is_symlink():
            raise ValueError('Invalid bundle path: ' + name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Missing or changed bundle file: ' + name)
    return manifest


def run(bundle, suites=None, library_path=DEFAULT_LIBRARIES, verify_only=False):
    bundle = bundle.resolve()
    manifest = verify(bundle)
    suites = suites or list(manifest['suites'])
    if set(suites) - manifest['suites'].keys():
        raise ValueError('Requested suite is absent from this bundle')
    if verify_only:
        print(f"Verified {len(manifest['files'])} files; suites: {', '.join(suites)}")
        return 0
    if os.uname().machine not in ('aarch64', 'arm64'):
        raise ValueError('Execution requires an AArch64 target; --verify-only works on the build host')
    results = bundle/'runs'
    if any((results/suite).exists() for suite in suites):
        raise ValueError('Execution results already exist; use a fresh bundle copy for another run')
    results.mkdir(exist_ok=True)
    ggml = bundle/'runtime/ggml'
    libraries = str(ggml/'ggml/src') + ':' + library_path if ggml.exists() else library_path
    config = dict(library_path=libraries, cpu=0,
                  environment={'POLYGEIST_GGML_BUILD': str(ggml)} if ggml.exists() else {})
    save(results/'target-config.json', config)
    env = dict(os.environ, LL_EXECUTION_BUNDLE=str(bundle/'execution-bundle.json'),
               LL_ARTIFACT_CONFIG=str(results/'target-config.json'),
               POLYGEIST_ROOT=str(bundle/'third_party/Polygeist'), PYTHONDONTWRITEBYTECODE='1')
    statuses = read(results/'execution.json') if (results/'execution.json').exists() else {}
    for suite in suites:
        output = results/suite
        shutil.copytree(bundle/'builds'/suite, output)
        dependencies = []
        for name in manifest['executables']:
            if not name.startswith('builds/'+suite+'/'):
                continue
            check = subprocess.run(['ldd', str(bundle/name)], text=True, capture_output=True,
                                   env=dict(env, LD_LIBRARY_PATH=libraries), timeout=30)
            dependencies.append(dict(executable=name, returncode=check.returncode,
                                     output=check.stdout+check.stderr))
        save(output/'runtime-dependencies.json', dependencies)
        if any(d['returncode'] or 'not found' in d['output'] for d in dependencies):
            statuses[suite] = dict(returncode=1, error='Missing runtime libraries; see runtime-dependencies.json')
        else:
            cmd = [sys.executable, str(bundle/'reproduce.py'), suite, '--stage', 'run', '--output', str(output)]
            if suite == 'mfem':
                cmd += ['--cases', *sorted(p.parent.name for p in output.glob('*/build.json'))]
            started = time.monotonic()
            with (results/(suite+'.log')).open('w') as log:
                rc = subprocess.run(cmd, cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            statuses[suite] = dict(returncode=rc, seconds=time.monotonic()-started,
                                   log=suite+'.log')
        save(results/'execution.json', statuses)
        print(suite, json.dumps(statuses[suite]), flush=True)
    return int(any(statuses[s]['returncode'] for s in suites))


def remote(bundle, host, destination, results, jump_host=None, library_path=DEFAULT_LIBRARIES, suites=None,
           ssh_options=()):
    verify(bundle)
    if not host or host.startswith('-') or any(c.isspace() for c in host):
        raise ValueError('Use an SSH alias or user@host')
    if not destination.startswith('/') or destination == '/':
        raise ValueError('--remote-dir must be an absolute directory dedicated to this run')
    if results.exists():
        raise ValueError('Use a fresh local --results directory')
    # OpenSSH uses the first value: an explicit option can enable an existing
    # SSH_ASKPASS setup without putting credentials in the bundle or command.
    options = [item for option in ssh_options for item in ('-o', option)]
    options += ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
    if jump_host:
        options += ['-J', jump_host]
    ssh = ['ssh', *options, host]
    # Authenticate before creating a remote directory or transferring any files.
    subprocess.run([*ssh, 'true'], check=True)
    with tempfile.TemporaryDirectory(prefix='ll-transfer-') as tmp:
        archive = Path(tmp)/'bundle.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name in ['execution-bundle.json', *read(bundle/'execution-bundle.json')['files']]:
                tar.add(bundle/name, arcname=name, recursive=False)
        command = ('umask 077; mkdir -- '+shlex.quote(destination)+
                   ' && tar -xzf - -C '+shlex.quote(destination))
        with archive.open('rb') as stream:
            subprocess.run([*ssh, command], stdin=stream, check=True)
    args = ['python3', destination+'/target.py', 'run', '--bundle', destination,
            '--library-path', library_path]
    if suites:
        args += ['--suites', *suites]
    results.mkdir(parents=True)
    with (results/'remote.log').open('w') as log:
        rc = subprocess.run([*ssh, shlex.join(args)], stdout=log, stderr=subprocess.STDOUT).returncode
    # Retrieve diagnostics even when correctness, timing, or dependency checks failed.
    copied = subprocess.run(['scp', *options, '-r', host+':'+destination+'/runs', str(results)]).returncode
    save(results/'transport.json', dict(returncode=rc, retrieval_returncode=copied,
                                       remote_directory=destination))
    return rc or copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('pack', help='Copy only execution inputs from successful builds')
    p.add_argument('--build', action='append', required=True, metavar='SUITE=PATH')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ggml-build', type=Path)
    for action in ('run', 'remote'):
        p = sub.add_parser(action)
        p.add_argument('--bundle', type=Path, required=True)
        p.add_argument('--suites', choices=SUITES, nargs='+')
        p.add_argument('--library-path', default=DEFAULT_LIBRARIES)
        if action == 'run':
            p.add_argument('--verify-only', action='store_true')
        else:
            p.add_argument('--host', required=True, help='User-supplied SSH alias or user@host; uses existing SSH authentication')
            p.add_argument('--jump-host')
            p.add_argument('--ssh-option', action='append', default=[], metavar='KEY=VALUE',
                           help='OpenSSH option, e.g. BatchMode=no for an external SSH_ASKPASS helper')
            p.add_argument('--remote-dir', default='/tmp/looplibrarian-'+uuid.uuid4().hex[:12])
            p.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'pack':
        builds = {}
        for value in args.build:
            suite, separator, path = value.partition('=')
            if not separator or suite not in SUITES or suite in builds:
                parser.error('--build needs a distinct supported SUITE=PATH')
            builds[suite] = Path(path).resolve()
        manifest = pack(args.output, builds, args.ggml_build)
        print('Packaged', len(manifest['executables']), 'executables in', args.output)
        return 0
    if args.action == 'run':
        return run(args.bundle, args.suites, args.library_path, args.verify_only)
    return remote(args.bundle.resolve(), args.host, args.remote_dir, args.results.resolve(),
                  args.jump_host, args.library_path, args.suites, args.ssh_option)


if __name__ == '__main__':
    sys.exit(main())
