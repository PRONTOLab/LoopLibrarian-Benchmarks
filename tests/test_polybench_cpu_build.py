"""A completed compiler invocation must still produce a library replacement."""
import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/artifact'), str(ROOT/'scripts/correctness')]
spec = importlib.util.spec_from_file_location('polybench_cpu_driver', ROOT/'polybench/cpu/reproduce.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class CPUReplacementBuild(unittest.TestCase):
    def test_no_match_is_reported_before_runtime_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in ['polybench/sources/tiny.c', 'polybench/sources/tiny.h',
                             'polybench/runtime/polybench_repeat_driver.c',
                             'compiler/build/bin/cgeist', 'compiler/build/bin/polygeist-opt']:
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test input\n')
            out = root/'output'
            out.mkdir()
            args = argparse.Namespace(target='host', openblas_root=None, output=out,
                                      size='MINI', dtype='double', kernels='tiny',
                                      preserve_views='', memory_form='')
            seen = []

            def command(cmd, log, env):
                seen.append(cmd)
                if '--whole-program' in cmd:
                    work = out/'tiny/work'
                    work.mkdir()
                    (work/'matched.mlir').write_text('module {}\n')
                    self.assertEqual(env['POLYGEIST_MATCH_REGION_BUDGET'], '300')
                elif '-o' in cmd:
                    Path(cmd[cmd.index('-o')+1]).write_bytes(b'object')
                else:
                    self.fail('Unexpected relink after no match: '+str(cmd))
                return 0

            def output(*args, **kwargs):
                return 'test' if kwargs.get('text') else b'test'

            with mock.patch.multiple(driver, ROOT=root, COMPILER=root/'compiler',
                                     POLYBENCH=root/'polybench/sources'), \
                 mock.patch.object(driver, 'command', side_effect=command), \
                 mock.patch.object(driver.subprocess, 'check_output', side_effect=output), \
                 mock.patch.dict(driver.os.environ, {}, clear=True):
                result = driver.build(args)['kernels'][0]
            self.assertEqual(len(seen), 4)
            self.assertEqual(result['native_build_rc'], 0)
            self.assertEqual(result['raised_build_rc'], 1)
            self.assertEqual(result['calls'], [])
            self.assertIn('No imported library call', result['error'])
            self.assertNotIn('raised_sha256', result)


if __name__ == '__main__':
    unittest.main()
