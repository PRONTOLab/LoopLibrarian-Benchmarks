"""Paths for benchmark inputs and the pinned compiler dependency."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPILER = Path(os.environ.get('POLYGEIST_ROOT', ROOT / 'third_party/Polygeist')).resolve()
DEPENDENCIES = Path(os.environ.get('POLYGEIST_DEPENDENCY_ROOT', ROOT / 'third_party')).resolve()


def source_path(value):
    """Resolve current sources and source names retained in historical manifests."""
    value = str(value)
    old = '/home/arjaiswal/Polygeist/'
    if value.startswith(old):
        value = value[len(old):]
    for prefix, target in (
        ('results/archive/issues/aten_c_kernels/', 'aten/inputs/'),
        ('issues/aten_c_kernels/', 'aten/inputs/'),
        ('results/archive/issues/mfem_c_kernels/', 'mfem/inputs/'),
        ('issues/mfem_c_kernels/', 'mfem/inputs/'),
        ('inputs/mfem/', 'mfem/inputs/'),
        ('inputs/llama/', 'llama/inputs/'),
        ('tools/cgeist/Test/polybench/', 'polybench/sources/'),
    ):
        if value.startswith(prefix):
            return ROOT / (target + value[len(prefix):])
    if value.startswith('issues/'):
        value = 'results/archive/' + value
    return ROOT / value
