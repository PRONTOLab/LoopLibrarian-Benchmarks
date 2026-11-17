"""Exercise the actual generated-harness comparator on finite and special values."""
import ctypes
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


class PublicationComparator(unittest.TestCase):
    def test_generated_checker_exits_on_publication_errors_and_preserves_guards(self):
        # Host memory stands in for CUDA allocations so the actual generated
        # checker and its exit status can be exercised without a GPU.
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'scripts/correctness'))
        from aten_fixture_batch import harness, correctness_summary
        cuda_stub='''
typedef int cudaError_t;
enum { cudaSuccess=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2 };
static const char *cudaGetErrorString(int code) { return "test"; }
static int cudaMalloc(void **p,size_t n) { *p=malloc(n); return !*p; }
static int cudaMemcpy(void *d,const void *s,size_t n,int kind) { memcpy(d,s,n); return 0; }
static int cudaDeviceSynchronize(void) { return 0; }
'''
        for dtype in ('float','double'):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                path=Path(directory)
                cfg=dict(dims={},args=[('out','ptr','1',True,'output')])
                if dtype=='double':cfg['dtype']='double'
                source=harness('test_kernel',cfg)
                source+='''
void test_kernel(TYPE *out) {
  puts("POLYGEIST_V2_CALL test");
  out[0]=100+atof(getenv("TEST_ERROR"));
  if(atoi(getenv("TEST_GUARD"))) ((unsigned char*)out)[-1]=0;
}
void native_reference(TYPE *out) { out[0]=100; }
void native_fixture(TYPE **out,const TYPE *parameters,const size_t *sizes,int n) { out[0][0]=100; }
'''.replace('TYPE',dtype)
                (path/'cuda_runtime_api.h').write_text(cuda_stub)
                (path/'check.c').write_text(source)
                subprocess.run(['cc','-O2','-fno-fast-math','-I',str(path),str(path/'check.c'),
                                '-lm','-o',str(path/'check')],check=True,capture_output=True)
                for error,guard,expected_rc in [('0.1','0',0),('1','0',3),('0.1','1',3)]:
                    result=subprocess.run([str(path/'check')],capture_output=True,text=True,
                        env=dict(os.environ,TEST_ERROR=error,TEST_GUARD=guard))
                    self.assertEqual(result.returncode,expected_rc,result.stdout)
                    parsed=correctness_summary(result.returncode,result.stdout,1,2)
                    self.assertEqual(parsed['passed'],expected_rc==0)
                    self.assertEqual(parsed['publication_passed'],expected_rc==0)
                    self.assertEqual(parsed['correctness_policy'],'publication')
                    self.assertNotIn('PUBLICATION_CHECK',result.stdout)

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
