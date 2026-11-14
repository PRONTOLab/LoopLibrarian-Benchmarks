#!/usr/bin/env python3
"""Configure the container's host tools; target SDK paths are supplied at run time."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--jobs', type=int, default=8)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
config = json.loads((root / 'config.example.json').read_text())
config['jobs'] = a.jobs
for key in ('kernelfarer_root', 'ginsbach_root', 'speq_root'):
    config[key] = str(root / config[key])
# Avoid placeholder SDK paths overriding a reviewer's environment.
config.pop('openblas_root', None)
config['environment'] = {'POLYGEIST_GGML_BUILD': str(root / 'build/ggml-aarch64')}
(root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
