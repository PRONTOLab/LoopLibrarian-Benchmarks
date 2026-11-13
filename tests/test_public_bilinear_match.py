"""Public interpolation reference matches only after common source raising.

The input fixture is read unchanged. No matching result is manufactured from
its name, and a reference with different coordinate divisors must not be accepted. Function
renaming must not affect the result.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts/correctness'),
               str(Path(os.environ.get('POLYGEIST_ROOT', Path(__file__).resolve().parents[1]/'third_party/Polygeist')) / 'scripts/correctness/lib_v2')]
import build_lib_v2
import match_v2
import recipe

REPO = Path(__file__).resolve().parents[1]


class PublicBilinearMatch(unittest.TestCase):
    def test_dynamic_reference_renaming_and_wrong_divisor(self):
        name = 'opencv_bilinear2x_f32'
        spec = next(s for s in build_lib_v2.load_specs() if s.name == name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root/(name+'.c')
            wrapper.write_text('#define NETLIB_SPEC '+name+'\n#include "'+
                               str(spec.ref_c)+'"\n')
            sources = [('library', wrapper, name),
                       ('app', REPO/'aten/inputs/aten_upsample_bilinear2d.c',
                        'aten_upsample_bilinear2d')]
            for stem, source, function in sources:
                result = recipe.raise_c(recipe.PRODUCTION, str(source), function,
                                        str(root/stem), [], timeout=120)
                for stage in ('cgeist', 'raise', 'debuf'):
                    self.assertTrue(result[stage].ok, (stage, result[stage].stderr))
            application = (root/'app_debuf.mlir').read_text()
            self.assertNotIn('affine.for', application)
            self.assertNotIn('scf.for', application)
            with patch.object(build_lib_v2, 'RAISED', root):
                definition, metadata = build_lib_v2.to_defn(
                    (root/'library_debuf.mlir').read_text(), spec)
            text = ('module {\n// lib_v2-meta: '+json.dumps(metadata)+'\n'+
                    definition+'\n}\n')
            library = root/'library.mlir'
            library.write_text(text)
            with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
                _, report, diagnostic = match_v2.run(application, library, region=False,
                                                     diag=True)
                self.assertEqual([item['spec'] for item in report], [name])
                self.assertEqual(diagnostic['residual_generics'], [])
                renamed = application.replace('@aten_upsample_bilinear2d', '@arbitrary_function_name')
                _, renamed_report, _ = match_v2.run(renamed, library, region=False)
                self.assertEqual([item['spec'] for item in renamed_report], [name])

                wrong = text.replace('arith.constant 2 : i32', 'arith.constant 3 : i32')
                self.assertNotEqual(text, wrong)
                library.write_text(wrong)
                _, report, _ = match_v2.run(application, library, region=False)
                self.assertEqual(report, [])


if __name__ == '__main__':
    unittest.main()
