"""Sparse index inputs must describe valid arrays in every value test mode."""
import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/correctness'))
import aten_fixture_batch as B


class SparseMetadata(unittest.TestCase):
    def test_signature_inference_retains_recorded_sparse_roles(self):
        cfg = B.configuration('aten_sparse_addmv_bsr_cpu')
        roles = {a[0]:a[4] for a in cfg['args']}
        self.assertEqual(roles['ptr'],'csr_rowptr')
        self.assertEqual(roles['col'],'csr_col')
        text = B.harness('aten_sparse_addmv_bsr_cpu',cfg)
        self.assertIn('if(array==0){*value=',text)
        self.assertIn('if(array==1){*value=',text)

    def test_balanced_rows_and_bounded_columns(self):
        # Include uneven row lengths and nnz smaller than the row count.
        for rows, cols, nnz in ((7, 11, 23), (16, 16, 64), (4096, 4096, 262144), (7, 11, 3)):
            with self.subTest(rows=rows, cols=cols, nnz=nnz):
                cfg = dict(dims=dict(R=rows, C=cols, N=nnz), args=[
                    ('ptr','iptr','R+1',False,'csr_rowptr'),
                    ('col','iptr','N',False,'csr_col'),
                    ('out','iptr','R',True,'normal')])
                text = B.harness('probe', cfg)
                body = text.split('static int initialize_integer_metadata(',1)[1].split('\nvoid probe(',1)[0]
                source = '#include <stdint.h>\n#include <stddef.h>\nint initialize_integer_metadata('+body
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp)
                    (p/'metadata.c').write_text(source)
                    subprocess.run(['gcc','-O2','-shared','-fPIC',str(p/'metadata.c'),'-o',str(p/'metadata.so')],
                                   check=True,capture_output=True)
                    call = ctypes.CDLL(str(p/'metadata.so')).initialize_integer_metadata
                    call.argtypes = [ctypes.c_int,ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32)]
                    def value(array,index):
                        v = ctypes.c_int32(-2147483648)
                        self.assertEqual(call(array,index,ctypes.byref(v)),1)
                        return v.value
                    offsets = [value(0,i) for i in range(rows+1)]
                    self.assertEqual(offsets[0],0)
                    self.assertEqual(offsets[-1],nnz)
                    self.assertEqual(offsets,sorted(offsets))
                    for i in range(nnz):
                        self.assertTrue(0 <= value(1,i) < cols)
                    sentinel = ctypes.c_int32(-7)
                    self.assertEqual(call(2,0,ctypes.byref(sentinel)),0)
                    self.assertEqual(sentinel.value,-7)


if __name__ == '__main__':
    unittest.main()
