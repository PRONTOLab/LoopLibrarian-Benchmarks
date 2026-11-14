"""Execute generated argument forwarding against independent C ABI callees."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1] / 'scripts/correctness'
sys.path.insert(0,str(HERE))
import aten_fixture_batch as B
from ll_paths import DEPENDENCIES


class FixtureScalarABI(unittest.TestCase):
    def execute(self, configuration, callee):
        source=B.harness('probe',configuration).replace('int main(', 'static int unused_main(')
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'check.c').write_text(source+'\n'+callee)
            result = subprocess.run(['gcc','-O2','-ffunction-sections','-fdata-sections',
                '-I'+str(DEPENDENCIES/'cuda_headers/cudart/include'),
                '-I'+str(DEPENDENCIES/'cuda_headers/cuda_crt/include'),
                str(d/'check.c'),'-Wl,--gc-sections','-lm','-o',str(d/'check')],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(d/'check')],check=True)

    def test_integer_and_float_arguments_keep_separate_abi_types(self):
        self.assertIn('iscalar',[a[1] for a in B.configuration('aten_allany_dims_cpu')['args']])
        cfg={'dims':{},'args':[('x','ptr','1',False,''),('mode','iscalar','2',False,''),
             ('alpha','scalar','0.5',False,''),('y','ptr','1',True,'')]}
        self.execute(cfg,'''void probe(float *x,int32_t mode,float alpha,float *y){y[0]=mode+alpha+x[0];}
int main(){float x=1,y=0;float *p[]={&x,&y};
parameters[0]=-3;parameters[1]=0.25f;invoke(p);if(y!=-1.75f)return 1;
parameters[0]=0;invoke(p);return y!=1.25f;}''')

    def test_double_arguments_preserve_precision(self):
        cfg={'dims':{},'dtype':'double','args':[('x','ptr','1',False,''),
             ('alpha','scalar','0.5',False,''),('y','ptr','1',True,'')]}
        self.execute(cfg,'''void probe(double *x,double alpha,double *y){y[0]=x[0]+alpha;}
int main(){double x=1,y=0;double *p[]={&x,&y};
parameters[0]=0x1p-40;invoke(p);return y!=1.+0x1p-40;}''')


if __name__=='__main__':unittest.main()
