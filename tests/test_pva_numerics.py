import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pva_driver', ROOT / 'pva/reproduce.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class PvaNumerics(unittest.TestCase):
    def test_recorded_approximation_still_requires_safe_vendor_execution(self):
        case=next(c for c in driver.CASES if c['name']=='bilateral_u8')
        log=('pvaBilateralFilterSubmit\nIMAGE_RAISED,NUMERICS_RECORDED,mismatches=10,'
             'max_error=73,input_unchanged=1,border_unchanged=1,guards_ok=1,input=random\n')
        self.assertTrue(driver.check_output(case,'random',0,log)['correctness'])
        for bad in (log.replace('guards_ok=1','guards_ok=0'),
                    log.replace('input_unchanged=1','input_unchanged=0'),
                    log.replace('pvaBilateralFilterSubmit','unrelated')):
            self.assertFalse(driver.check_output(case,'random',0,bad)['correctness'])
        self.assertFalse(driver.check_output(case,'random',1,log)['correctness'])

    def test_vendor_differences_are_retained_without_claiming_a_match(self):
        lines=[]
        for family in ('BoxFilter','BilateralFilter','HistogramEqualization'):
            for pattern in ('original_filter','random','impulse','constant','quantization_extremes'):
                lines.append(f'COMPARE,{family},{pattern},count=49152,mismatches=200,'
                             'max_error=73,guardbad=0,inputbad=0')
        log='\n'.join(lines)
        result=driver.vendor_comparison_result(0,log)
        self.assertTrue(result['execution_complete'])
        self.assertEqual(result['comparisons'][-1]['max_error'],73)
        self.assertFalse(driver.vendor_comparison_result(1,log)['execution_complete'])
        self.assertFalse(driver.vendor_comparison_result(0,'\n'.join(lines[:-1]))['execution_complete'])
        self.assertFalse(driver.vendor_comparison_result(0,log.replace('guardbad=0','guardbad=1'))['execution_complete'])

    def test_build_reports_missing_sdk_before_compilation(self):
        with patch.object(driver, 'environment', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'PVA_OPERATOR_INCLUDE.*CUPVA_SDK_ROOT'):
                driver.check_build_dependencies()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fields={'PVA_OPERATOR_INCLUDE':'OpBoxFilter.h','NVCV_INCLUDE':'nvcv/Tensor.h',
                    'CUPVA_SDK_ROOT':'include/cupva_host.h','PVA_LIB_STAGE':'libpva_operator.so'}
            env={}
            for key,relative in fields.items():
                base=root/key;path=base/relative
                path.parent.mkdir(parents=True,exist_ok=True);path.touch()
                env[key]=str(base)
            with patch.object(driver, 'environment', return_value=env):
                driver.check_build_dependencies()
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
