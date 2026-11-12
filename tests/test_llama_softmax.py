"""The Llama fixture matches the FP32 library and rejects changed semantics."""
import os
import re
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
BENCHMARKS = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('POLYGEIST_ROOT', BENCHMARKS/'third_party/Polygeist')).resolve()
sys.path[:0] = [str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import match_v2 as M
import recipe as R

class ImportedSoftmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = ROOT / 'generic_solver/kernel_library_v3.mlir'
        source = BENCHMARKS / 'inputs/extracted/cnn-extracted/llama_forward_ops.c'
        if not source.is_file() or '@aten_softmax_f32(' not in cls.lib.read_text():
            raise unittest.SkipTest('run the FP32 reference import first')
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        steps = R.raise_c(R.PRODUCTION, str(source), 'kernel_llama_attention_softmax',
                          str(Path(cls.temp.name) / 'application'), [])
        if not steps.get('debuf') or not steps['debuf'].ok:
            raise AssertionError('fresh softmax raising failed: ' + repr(steps))
        cls.app = Path(steps['paths']['debuf'])

    def matches(self, text):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            return M.run(text, self.lib, region=False, allowed_specs={'aten_softmax_f32'})[1]

    def test_complete_fp32_reference_matches(self):
        matches = self.matches(self.app.read_text())
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(matches[0]['generics']), 3)

    def test_different_precision_does_not_match(self):
        self.assertFalse(self.matches(self.app.read_text().replace('f32', 'f64')))

    def test_inconsistent_reduction_bound_does_not_match(self):
        self.assertFalse(self.matches(self.app.read_text().replace('constant 31 : index',
                                                                  'constant 30 : index')))

    def test_incorrect_computation_does_not_match(self):
        self.assertFalse(self.matches(self.app.read_text().replace('arith.divf', 'arith.mulf')))

    def test_incorrect_producer_result_does_not_match(self):
        text = self.app.read_text()
        result = re.search(r'(%[\w.-]+)#1', text)
        self.assertIsNotNone(result)
        self.assertFalse(self.matches(text.replace(result[0], result[1]+'#0')))

    def test_changed_reduction_seed_does_not_match(self):
        self.assertFalse(self.matches(self.app.read_text().replace('0.000000e+00 : f32',
                                                                  '1.000000e+00 : f32')))


if __name__ == '__main__':
    unittest.main()
