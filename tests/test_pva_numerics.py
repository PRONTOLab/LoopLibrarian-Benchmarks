import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pva_driver', ROOT / 'pva/reproduce.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class PvaNumerics(unittest.TestCase):
    def check(self, name='box_u8', error=3, **kwargs):
        case = next(c for c in driver.CASES if c['name'] == name)
        log = (case['vendor'] + '\nFILTER_RAISED,box,u8,WITHIN_TOLERANCE,'
               'mismatches=10,max_error=' + str(error) + ',tolerance=3,'
               'input_unchanged=1,border_unchanged=1,guards_ok=1,input=quantization\n')
        if 'replace' in kwargs:
            log = log.replace(*kwargs['replace'])
        return driver.check_output(case, 'quantization', kwargs.get('rc', 0), log)

    def test_quantized_case_accepts_only_declared_bound(self):
        self.assertTrue(self.check()['correctness'])
        self.assertFalse(self.check(error=4)['correctness'])

    def test_exact_case_does_not_inherit_approximation_budget(self):
        self.assertFalse(self.check(name='gaussian_u8', error=1)['correctness'])

    def test_tolerance_does_not_excuse_memory_or_border_errors(self):
        for key in ('input_unchanged', 'border_unchanged', 'guards_ok'):
            with self.subTest(key=key):
                self.assertFalse(self.check(replace=(key+'=1', key+'=0'))['correctness'])

    def test_requires_requested_input_and_observed_vendor_call(self):
        self.assertFalse(self.check(replace=('input=quantization', 'input=original'))['correctness'])
        self.assertFalse(self.check(replace=('pvaBoxFilterSubmit', 'unrelated'))['correctness'])
        self.assertFalse(self.check(rc=1)['correctness'])


if __name__ == '__main__':
    unittest.main()
