"""Run benchmarks on the current machine and stage files locally."""
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess

from common import setting


def require_aarch64():
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError(
            'These executables require an AArch64 machine. Build with --stage build, '
            'then run with --stage run on your own compatible target; see docs/reproduction.md.')


def run(command, timeout=300):
    process = subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, output
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return 124, output + '\nBenchmark command timed out.\n'


def stage(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return 0


def runtime_env():
    libraries = setting('library_path', '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu')
    return 'LD_LIBRARY_PATH=' + shlex.quote(libraries) + ' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
