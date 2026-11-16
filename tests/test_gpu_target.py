"""Keep host compilation, CUDA architecture and execution targets consistent."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/artifact'))
import common
import gpu_target as gpu


class GPUTarget(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.config = patch.object(common, 'CONFIG', {})
        self.env.start()
        self.config.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.config.stop)

    def test_default_build_is_native(self):
        self.assertEqual(gpu.compiler_target(), 'cuda')
        self.assertEqual(gpu.cc(), 'gcc')
        self.assertEqual(gpu.nvcc_flags(), ['-arch=sm_80', '-ccbin', 'g++'])
        self.assertIn('x86_64-linux', str(gpu.cuda_root()))
        self.assertNotIn('-DHAVE_SVE256_CPU_DEFINITION', gpu.torch_compile_flags())

    def test_jetson_is_explicit_and_uses_cross_compiler(self):
        os.environ['LL_GPU_TARGET'] = 'jetson'
        self.assertEqual(gpu.compiler_target(), 'jetson')
        self.assertEqual(gpu.cc(), 'aarch64-linux-gnu-gcc')
        self.assertIn('-DHAVE_SVE256_CPU_DEFINITION', gpu.torch_compile_flags())
        self.assertEqual(gpu.nvcc_flags(), ['-arch=sm_87', '-ccbin', 'aarch64-linux-gnu-g++', '-target-dir', 'sbsa-linux'])
        with patch.object(gpu.platform, 'machine', return_value='x86_64'):
            with self.assertRaisesRegex(RuntimeError, 'cannot run'):
                gpu.require_execution()

    def test_native_execution_rejects_arm(self):
        with patch.object(gpu.platform, 'machine', return_value='aarch64'):
            with self.assertRaisesRegex(RuntimeError, 'cannot run'):
                gpu.require_execution()

    def test_architecture_override_reaches_all_builds(self):
        os.environ['LL_GPU_ARCH'] = 'sm_90'
        env = common.environment()
        self.assertEqual(env['POLYGEIST_GPU_ARCH'], 'sm_90')
        self.assertEqual(env['POLYGEIST_GPU_TARGET'], 'x86_64')
        self.assertIn('-arch=sm_90', gpu.nvcc_flags())

    def test_wrong_architecture_sdk_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lib = Path(directory)/'lib'
            lib.mkdir()
            header = bytearray(20)
            header[:4] = b'\x7fELF'
            header[18:20] = (183).to_bytes(2, 'little')
            (lib/'libcudart.so').write_bytes(header)
            os.environ['LL_CUDA_TARGET'] = directory
            with self.assertRaisesRegex(RuntimeError, 'architecture does not match'):
                gpu.validate_toolkit()
            os.environ['LL_GPU_TARGET'] = 'jetson'
            gpu.validate_toolkit()

    def test_compiler_shell_agrees_with_target(self):
        compiler = Path(os.environ.get('POLYGEIST_ROOT', common.COMPILER))
        helper = compiler/'scripts/correctness/gpu_target.sh'
        if not helper.exists():
            self.skipTest('Compiler dependency does not yet include the target selector')
        for target, triple, cc in [('x86_64', 'x86_64-linux-gnu', 'gcc'),
                                   ('jetson', 'aarch64-linux-gnu', 'aarch64-linux-gnu-gcc')]:
            result = subprocess.check_output(['bash', '-c',
                'source "$1"; configure_gpu_target "$2"; printf "%s\\n" "$GPU_TRIPLE" "$GPU_CC" "${GPU_NVCC_ARGS[*]}"',
                'bash', str(helper), target], text=True).splitlines()
            self.assertEqual(result[:2], [triple, cc])
            self.assertEqual('sbsa-linux' in result[2], target == 'jetson')


if __name__ == '__main__':
    unittest.main()
