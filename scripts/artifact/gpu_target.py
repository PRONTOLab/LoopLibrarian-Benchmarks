"""CUDA build settings shared by all GPU experiments."""
import os
from pathlib import Path
import platform
import re


def name():
    from common import setting
    value = setting('gpu_target', os.environ.get('POLYGEIST_GPU_TARGET', 'x86_64'))
    if value not in ('x86_64', 'jetson'):
        raise ValueError('gpu_target must be x86_64 or jetson')
    return value


def compiler_target():
    return 'jetson' if name() == 'jetson' else 'cuda'


def cc():
    return os.environ.get('AARCH64_CC', 'aarch64-linux-gnu-gcc') if name() == 'jetson' else os.environ.get('POLYGEIST_HOST_CC', 'gcc')


def cxx():
    return os.environ.get('AARCH64_CXX', 'aarch64-linux-gnu-g++') if name() == 'jetson' else os.environ.get('POLYGEIST_HOST_CXX', 'g++')


def architecture():
    from common import setting
    value = str(setting('gpu_arch', os.environ.get('POLYGEIST_GPU_ARCH', 'sm_87' if name() == 'jetson' else 'sm_80')))
    if not re.fullmatch(r'sm_[0-9]+', value):
        raise ValueError('gpu_arch must name a CUDA architecture, for example sm_80')
    return value


def nvcc_flags():
    flags = ['-arch=' + architecture(), '-ccbin', cxx()]
    if name() == 'jetson':
        flags += ['-target-dir', 'sbsa-linux']
    return flags


def cuda_root():
    from common import setting
    default = '/usr/local/cuda/targets/' + ('sbsa-linux' if name() == 'jetson' else 'x86_64-linux')
    return Path(setting('cuda_target', os.environ.get('CUDA', default)))


def cuda_lib():
    root = cuda_root()
    return root / ('lib64' if not (root/'lib').exists() and (root/'lib64').exists() else 'lib')


def system_lib():
    return '/usr/lib/' + ('aarch64-linux-gnu' if name() == 'jetson' else 'x86_64-linux-gnu')


def library_path():
    from common import setting
    return setting('library_path', ':'.join([str(cuda_lib()), '/usr/local/cuda/lib64', system_lib(), os.environ.get('LD_LIBRARY_PATH', '')]))


def torch_root():
    root = os.environ.get('ATEN_NATIVE_TORCH_ROOT')
    if root:
        return Path(root)
    # Importing torch is unnecessary for matching-only runs.
    return Path('/opt/libtorch')


def torch_abi():
    from common import setting
    return str(setting('torch_cxx11_abi', os.environ.get('ATEN_TORCH_CXX11_ABI', '1' if name() == 'jetson' else '0')))


def torch_compile_flags():
    flags = ['-D_GLIBCXX_USE_CXX11_ABI=' + torch_abi()]
    if name() == 'jetson':
        flags.append('-DHAVE_SVE256_CPU_DEFINITION')
    return flags


def require_execution():
    expected = ('aarch64', 'arm64') if name() == 'jetson' else ('x86_64', 'amd64')
    if platform.machine().lower() not in expected:
        raise RuntimeError(f'{name()} executables cannot run on {platform.machine()}; select the matching --gpu-target or transfer the build to that target.')


def validate_toolkit():
    """Catch an ARM SDK selected for native builds before expensive matching."""
    library = cuda_lib()/'libcudart.so'
    if not library.is_file():
        raise RuntimeError(f'Missing CUDA development library: {library}. Use the GPU image or configure cuda_target for {name()}.')
    with library.open('rb') as stream:
        header = stream.read(20)
    expected = 183 if name() == 'jetson' else 62
    if header[:4] != b'\x7fELF' or int.from_bytes(header[18:20], 'little') != expected:
        raise RuntimeError(f'CUDA SDK architecture does not match {name()}: {library}')


def environment():
    from common import ROOT
    root = str(cuda_root())
    return dict(POLYGEIST_GPU_TARGET=name(), POLYGEIST_GPU_ARCH=architecture(),
                CUDA=root, POLYGEIST_CUDA_CROSS_ROOT=root,
                ATEN_NATIVE_CUDA_ROOT=root,
                POLYGEIST_GGML_BUILD=str(ROOT/'build'/('ggml-aarch64' if name() == 'jetson' else 'ggml-x86_64')))
