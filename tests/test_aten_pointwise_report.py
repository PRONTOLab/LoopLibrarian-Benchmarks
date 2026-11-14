"""Partial matches and failed comparisons must not increase the pass counts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pointwise_driver', ROOT/'aten/pointwise.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class PointwiseReport(unittest.TestCase):
    def test_verified_external_gpu_evidence_still_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); out = root/'current'; other = root/'other'
            for run in (out,other):
                (run/'matching/aten_exp2/rank1').mkdir(parents=True)
            (other/'gpu/aten_exp2').mkdir(parents=True)
            matched = dict(source_sha256='source',variants=[dict(emitted=['spec'],abi_ok=True,counts=dict(linalg=0,loops=0))])
            (out/'matching/aten_exp2/result.json').write_text(json.dumps(matched))
            (other/'gpu/aten_exp2/build.json').write_text(json.dumps(dict(
                rc=0,source_sha256='source',binary_sha256='binary',rank=1)))
            proof = dict(library_sha256='library',semantic_inputs={},configuration={},abi_sha256='abi')
            for run in (out,other):
                (run/'matching/aten_exp2/rank1/proof.json').write_text(json.dumps(proof))
            evidence = other/'gpu/gpu-results.json'
            evidence.write_text(json.dumps([dict(kernel='aten_exp2',binary_sha256='binary',
                fully_validated=True,timing=dict(passed=True,native_over_raised=2,samples=[{}]*10))]))
            driver.report(out,['aten_exp2'],extra_gpu_results=[evidence])
            summary = json.loads((out/'summary.json').read_text())['summary']
            self.assertEqual(summary['strict_gpu_passes'], 1)
            self.assertEqual(summary['timed'], 1)

    def test_rebuilt_binary_cannot_reuse_old_successful_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            folder = out/'matching/aten_exp2'
            folder.mkdir(parents=True)
            (folder/'result.json').write_text(json.dumps(dict(source_sha256='source',variants=[
                dict(emitted=['spec'],abi_ok=True,counts=dict(linalg=0,loops=0))])))
            folder = out/'gpu/aten_exp2'
            folder.mkdir(parents=True)
            (folder/'build.json').write_text(json.dumps(dict(rc=0,source_sha256='source',binary_sha256='new')))
            (out/'gpu/gpu-results.json').write_text(json.dumps([dict(kernel='aten_exp2',
                binary_sha256='old',fully_validated=True,timing=dict(passed=True,native_over_raised=10))]))
            driver.report(out,['aten_exp2'],scope='all')
            report = json.loads((out/'summary.json').read_text())
            self.assertEqual(report['summary']['strict_gpu_passes'], 0)
            self.assertEqual(report['summary']['timed'], 0)
            self.assertEqual(report['cases'][0]['execution_status'], 'stale_or_incomplete_gpu_evidence')

    def test_full_inventory_retains_missing_adapters_and_failed_builds(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(driver, 'NATIVE', {'exp2':'native'}):
            out = Path(temporary)
            for name in ['aten_exp2','aten_gather_cpu']:
                folder = out/'matching'/name
                folder.mkdir(parents=True)
                (folder/'result.json').write_text(json.dumps(dict(status='recognized',variants=[
                    dict(emitted=['spec'],abi_ok=True,counts=dict(linalg=0,loops=0))])))
            folder = out/'gpu/aten_exp2'
            folder.mkdir(parents=True)
            (folder/'build.json').write_text(json.dumps(dict(rc=1,error='missing SDK')))
            driver.report(out,['aten_exp2','aten_gather_cpu','aten_unmatched'],scope='all')
            report = json.loads((out/'summary.json').read_text())
            self.assertEqual(report['cohort'], 'all')
            self.assertEqual(report['summary']['candidates'], 3)
            self.assertEqual(report['summary']['complete_matches'], 2)
            self.assertEqual(report['summary']['complete_without_native_adapter'], 1)
            self.assertEqual(report['summary']['strict_gpu_passes'], 0)
            self.assertEqual([r['execution_status'] for r in report['cases']],
                             ['build_failed','native_adapter_unavailable','incomplete_match'])

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
                (d/'request.json').write_text(json.dumps(dict(numeric_model='strict',
                    baseline_complete_match='False' if remaining == 0 else 'True')))
            (out/'gpu').mkdir()
            (out/'gpu/gpu-results.json').write_text(json.dumps([
                dict(kernel='aten_exp2',fully_validated=False,
                     timing=dict(passed=True,native_over_raised=10))]))
            driver.report(out, ['aten_exp2','aten_expm1','aten_log2'])
            summary = json.loads((out/'summary.json').read_text())['summary']
            self.assertEqual(summary, dict(candidates=3, complete_matches=1,
                strict_gpu_passes=0,timed=0,baseline_complete_matches=1,new_complete_matches=1))
            rows = json.loads((out/'summary.json').read_text())['cases']
            self.assertEqual(rows[0]['numeric_model'], 'strict')
            self.assertEqual(rows[2]['numeric_model'], 'unknown')


if __name__ == '__main__':
    unittest.main()
