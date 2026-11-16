import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/artifact'))
spec = importlib.util.spec_from_file_location('mfem_matching', ROOT/'mfem/matching.py')
matching = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matching)


class MfemMatching(unittest.TestCase):
    def test_pointwise_calls_do_not_inflate_contraction_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            catalog = out/'compiler/generic_solver/lib_v3/manifest'
            catalog.mkdir(parents=True)
            (catalog/'library.json').write_text(json.dumps(dict(specs=[
                dict(name='matrix', target=dict(library='cuBLAS', call='cublasDgemm')),
                dict(name='scale', target=dict(library='Thrust', call='transform')),
                dict(name='update', target=dict(library='cuBLAS', call='cublasDaxpy'))])))
            rows = [dict(kernel='mass_apply_2d'), dict(kernel='pcg_application')]
            for name, emitted, abi in [('mass_apply_2d',['matrix','matrix','scale'],True),
                                       ('pcg_application',['update'],False)]:
                dest = out/'matching'/name/'full'
                dest.mkdir(parents=True)
                (dest/'result.json').write_text(json.dumps(dict(
                    raised=True, recognized=True, abi_ok=abi, emitted=emitted, error=None)))
            with patch.object(matching, 'OUT', out), patch.object(matching, 'COMPILER', out/'compiler'):
                report = matching.summarize(rows)
            self.assertEqual(report['current_contraction_regions'], 2)
            self.assertEqual(report['results'][0]['emitted_regions'], 3)
            self.assertEqual(report['results'][0]['paper_regions'], 3)
            self.assertEqual(report['results'][0]['contraction_count_difference'], -1)
            self.assertEqual(report['pcg_vector_calls'], [])
            self.assertFalse(report['source_site_mapping_verified'])
            self.assertEqual(report['by_family']['mass_apply']['paper_sites'], 8)


if __name__ == '__main__':
    unittest.main()
