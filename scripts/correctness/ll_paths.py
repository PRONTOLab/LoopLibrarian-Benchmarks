"""Separate benchmark data from the pinned compiler and its dependencies."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
COMPILER = Path(os.environ.get('POLYGEIST_ROOT', ROOT / 'third_party/Polygeist')).resolve()
DEPENDENCIES = Path(os.environ.get('POLYGEIST_DEPENDENCY_ROOT', ROOT / 'third_party')).resolve()
def source_path(value):
    value = str(value)
    old = '/home/arjaiswal/Polygeist/'
    if value.startswith(old): value = value[len(old):]
    if value.startswith('issues/'): value = 'results/archive/' + value
    value = value.replace('tools/cgeist/Test/polybench/', 'polybench/sources/')
    return ROOT / value
