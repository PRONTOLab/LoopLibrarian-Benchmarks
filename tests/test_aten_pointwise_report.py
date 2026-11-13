"""Partial matches and failed comparisons must not increase the pass counts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pointwise_driver', ROOT/'aten/pointwise.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class PointwiseReport(unittest.TestCase):
    def test_external_evidence_cannot_replace_a_failed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary); (out/'gpu').mkdir()
            failed = dict(kernel='aten_exp2',fully_validated=False)
            (out/'gpu/gpu-results.json').write_text(json.dumps([failed]))
            extra = out/'extra.json'
            extra.write_text(json.dumps([dict(failed,fully_validated=True)]))
            with self.assertRaisesRegex(ValueError,'Duplicate GPU evidence'):
                driver.report(out,['aten_exp2'],extra_gpu_results=[extra])

    def test_external_evidence_requires_the_same_library_and_numeric_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); out = root/'current'; other = root/'other'
            for run in (out,other):
                (run/'matching/aten_exp2/rank1').mkdir(parents=True)
            (other/'gpu/aten_exp2').mkdir(parents=True)
            (out/'matching/aten_exp2/result.json').write_text(json.dumps(dict(source_sha256='source')))
            (other/'gpu/aten_exp2/build.json').write_text(json.dumps(dict(
                source_sha256='source',binary_sha256='binary',rank=1)))
            proof=dict(library_sha256='library',semantic_inputs={},configuration={},abi_sha256='abi')
            (out/'matching/aten_exp2/rank1/proof.json').write_text(json.dumps(proof))
            evidence=other/'gpu/gpu-results.json'
            evidence.write_text(json.dumps([dict(kernel='aten_exp2',binary_sha256='binary',fully_validated=True)]))
            for changed in (dict(proof,library_sha256='different'),
                            dict(proof,configuration={'numeric_model':'relaxed'})):
                with self.subTest(proof=changed):
                    (other/'matching/aten_exp2/rank1/proof.json').write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError,'different compiler/specification inputs'):
                        driver.report(out,['aten_exp2'],extra_gpu_results=[evidence])

    def test_partial_match_and_failed_gpu_timing_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for name, remaining in [('aten_exp2',0), ('aten_expm1',1)]:
                d = out/'matching'/name; d.mkdir(parents=True)
                (d/'result.json').write_text(json.dumps(dict(status='recognized', variants=[
                    dict(emitted=['imported_reference'],abi_ok=True,
                         counts=dict(linalg=remaining,loops=0))])))
                (d/'request.json').write_text(json.dumps(dict(numeric_model='strict')))
            (out/'gpu').mkdir()
            (out/'gpu/gpu-results.json').write_text(json.dumps([
                dict(kernel='aten_exp2',fully_validated=False,
                     timing=dict(passed=True,native_over_raised=10))]))
            driver.report(out, ['aten_exp2','aten_expm1','aten_log2'])
            summary = json.loads((out/'summary.json').read_text())['summary']
            self.assertEqual(summary, dict(candidates=3, complete_matches=1,
                                           strict_gpu_passes=0,timed=0))
            rows = json.loads((out/'summary.json').read_text())['cases']
            self.assertEqual(rows[0]['numeric_model'], 'strict')
            self.assertEqual(rows[2]['numeric_model'], 'unknown')


if __name__ == '__main__':
    unittest.main()
