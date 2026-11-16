#!/usr/bin/env python3
"""Configure the installed x86 CUDA libraries without host-specific paths."""
import argparse
import json
from pathlib import Path
import sysconfig
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu-arch', default='80')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
site = Path(sysconfig.get_paths()['purelib'])
torch_root = Path(torch.__file__).parent
cutensor = site/'cutensor'
cudnn = site/'nvidia/cudnn'
if not (cutensor/'include/cutensor.h').is_file() or not (cudnn/'include/cudnn.h').is_file():
    raise RuntimeError('Missing cuDNN or cuTENSOR development files')
# The shared runtime links cuDNN by its unversioned development name.
cudnn_link = cudnn/'lib/libcudnn.so'
if not cudnn_link.exists():
    cudnn_link.symlink_to('libcudnn.so.9')
libraries = [torch_root/'lib', *sorted((site/'nvidia').glob('*/lib')),
             cutensor/'lib', Path('/usr/local/cuda/lib64'), Path('/usr/lib/x86_64-linux-gnu')]
config = json.loads((root/'config.json').read_text())
config.update(gpu_target='x86_64', gpu_arch='sm_'+a.gpu_arch,
              nvcc='/usr/local/cuda/bin/nvcc',
              cuda_target='/usr/local/cuda/targets/x86_64-linux',
              library_path=':'.join(map(str, libraries)),
              torch_cxx11_abi=int(torch._C._GLIBCXX_USE_CXX11_ABI))
config['environment'] = dict(
    POLYGEIST_GGML_BUILD=str(root/'build/ggml-x86_64'),
    ATEN_NATIVE_TORCH_ROOT=str(torch_root),
    ATEN_TORCH_CXX11_ABI=str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
    POLYGEIST_CUDNN_INCLUDE=str(cudnn/'include'),
    POLYGEIST_CUDNN_LIBDIR=str(cudnn/'lib'),
    POLYGEIST_CUTENSOR_ROOT=str(cutensor),
    LD_LIBRARY_PATH=config['library_path'])
(root/'config.json').write_text(json.dumps(config, indent=2)+'\n')
