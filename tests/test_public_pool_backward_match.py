"""A public pooling reference must prove the complete update, not its name."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
COMPILER=Path(os.environ.get('POLYGEIST_ROOT', ROOT/'third_party/Polygeist'))
sys.path[:0]=[str(COMPILER/'scripts/correctness'),str(COMPILER/'scripts/correctness/lib_v2')]
import recipe
import spec_match


class PoolBackwardReference(unittest.TestCase):
    def test_whole_scatter_and_wrong_divisor(self):
        source=(ROOT/'aten/inputs/aten_avg_pool2d_backward_cpu.c').read_text()
        name='unrelated_function_name'
        source=source.replace('aten_avg_pool2d_backward_cpu',name)
        wrong=source.replace('(float)((e0-s0)*(e1-s1))','5.0f')
        self.assertNotEqual(source,wrong)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for label,text in [('correct',source),('wrong',wrong)]:
                path=root/(label+'.c');path.write_text(text)
                steps=recipe.raise_c(recipe.PRODUCTION,str(path),name,str(root/label),[],120)
                self.assertTrue(all(steps[s].ok for s in ['cgeist','raise','debuf']))
                with patch.dict(os.environ,POLYGEIST_NUMERIC_MODEL='relaxed'):
                    matched,proof=spec_match.match((root/(label+'_debuf.mlir')).read_text(),
                        backend='cuda',region=False,only=['aten_avg_pool2d_backward_k2_f32'])
                if label=='correct':
                    self.assertEqual(proof['emitted'],['aten_avg_pool2d_backward_k2_f32'])
                    self.assertNotIn('linalg.generic',matched)
                    self.assertNotIn('affine.for',matched)
                else:
                    self.assertEqual(proof['emitted'],[])


if __name__=='__main__': unittest.main()
