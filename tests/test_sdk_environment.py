"""Configured SDK locations must reach every compiler wrapper."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sdk_common', ROOT / 'scripts/artifact/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class SDKEnvironment(unittest.TestCase):
    def test_configured_toolkit_reaches_cross_linker_and_nvcc(self):
        config = {'cuda_target': '/sdk/cuda/targets/sbsa-linux',
                  'nvcc': '/sdk/cuda/bin/nvcc'}
        with patch.dict(os.environ, {}, clear=True), patch.object(common, 'CONFIG', config):
            env = common.environment()
        self.assertEqual(env['CUDA'], config['cuda_target'])
        self.assertEqual(env['POLYGEIST_CUDA_CROSS_ROOT'], config['cuda_target'])
        self.assertEqual(env['POLYGEIST_NVCC'], config['nvcc'])
        self.assertEqual(env['POLYGEIST_REAL_NVCC'], config['nvcc'])

    def test_explicit_overrides_remain_available(self):
        config = {'cuda_target': '/sdk/cuda/targets/sbsa-linux',
                  'nvcc': '/sdk/cuda/bin/nvcc',
                  'environment': {'POLYGEIST_REAL_NVCC': '/custom/nvcc'}}
        with patch.dict(os.environ, {}, clear=True), patch.object(common, 'CONFIG', config):
            env = common.environment(CUDA='/alternate/target')
        self.assertEqual(env['POLYGEIST_REAL_NVCC'], '/custom/nvcc')
        self.assertEqual(env['CUDA'], '/alternate/target')


if __name__ == '__main__':
    unittest.main()
