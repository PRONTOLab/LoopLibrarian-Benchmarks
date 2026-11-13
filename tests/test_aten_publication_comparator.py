"""Exercise the actual generated-harness comparator on finite and special values."""
import ctypes
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


class PublicationComparator(unittest.TestCase):
    def test_float_policy_preserves_integer_exactness_and_nonfinite_checks(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'scripts/correctness/aten_fixture_batch_harness.c').read_text()
        function=re.search(r'static int publication_equal_slot\(.*?\n}',source,re.S).group()
        function=function.replace('static int publication_equal_slot','int publication_equal_slot',1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'check.c').write_text('#include <math.h>\n#include <string.h>\n'
                'static int integer_arrays[]={0,1};\n'
                'static int element_bytes[]={sizeof(float),sizeof(int)};\n'+function+'\n')
            subprocess.run(['cc','-shared','-fPIC','-O2','-fno-fast-math',str(path/'check.c'),
                            '-lm','-o',str(path/'check.so')],check=True,capture_output=True)
            lib=ctypes.CDLL(str(path/'check.so'))
            equal=lib.publication_equal_slot
            equal.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_double,ctypes.c_int]
            equal.restype=ctypes.c_int
            def check(got,expected,finite=True,integer=False):
                ctype=ctypes.c_int if integer else ctypes.c_float
                a,b=ctype(got),ctype(expected)
                return bool(equal(int(integer),ctypes.byref(a),ctypes.byref(b),0.002,int(finite)))
            self.assertTrue(check(100.1,100))
            self.assertFalse(check(101,100))
            self.assertTrue(check(-0.0,0.0))
            self.assertFalse(check(float('nan'),float('nan')))
            self.assertTrue(check(float('nan'),float('nan'),finite=False))
            self.assertFalse(check(float('inf'),float('inf')))
            self.assertTrue(check(float('inf'),float('inf'),finite=False))
            self.assertFalse(check(1001,1000,integer=True))


if __name__=='__main__':unittest.main()
