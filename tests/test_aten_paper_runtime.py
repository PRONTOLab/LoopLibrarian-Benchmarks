import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/correctness'))
import aten_paper_runtime as R


class RuntimeSelection(unittest.TestCase):
    def test_resume_preserves_correctness_policy_and_input_scope(self):
        saved = dict(binary_sha256='same', cudnn_plan_cache=True,
                     gpu_workspace_cache=True, correctness={'paper': {}},
                     timing={'correctness_policy': 'publication'})
        built = dict(binary_sha256='same', rc=0)
        args = SimpleNamespace(timing_correctness='publication',
                               cudnn_plan_cache=True, gpu_workspace_cache=True)
        self.assertTrue(R.reusable_runtime(saved, built, args, False))
        self.assertFalse(R.reusable_runtime(saved, built, args, True))
        args.timing_correctness = 'strict'
        self.assertFalse(R.reusable_runtime(saved, built, args, False))
        saved['timing'] = {}
        self.assertTrue(R.reusable_runtime(saved, built, args, False))
        saved['timing_correctness'] = 'publication'
        self.assertFalse(R.reusable_runtime(saved, built, args, False))

    def test_csv_preserves_unmatched_and_failed_comparisons_without_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'aten').mkdir()
            (root/'aten/paper_timings.csv').write_text(
                'kernel,raised_us,native_us,timing_protocol\n'
                'aten_missing,1,2,minimum of 5 after 5 warmups\n'
                'aten_failed,1,2,minimum of 5 after 5 warmups\n')
            campaign=root/'matching';proof=campaign/'aten_failed/rank0'
            proof.mkdir(parents=True)
            (proof/'proof.json').write_text(json.dumps(dict(configuration={},emitted=['public_ref'])))
            output=root/'gpu';case=output/'aten_failed';case.mkdir(parents=True)
            (case/'build.json').write_text(json.dumps(dict(rc=0)))
            (case/'gpu-result.json').write_text(json.dumps(dict(fully_validated=False,
                correctness=dict(paper=dict(raised_passed=False,native_passed=False,
                                           raised_matches_native=True,passed=False)),
                timing=dict(passed=False))))
            cases=[dict(kernel='aten_missing',complete_match=False),
                   dict(kernel='aten_failed',complete_match=True,campaign=str(campaign),rank=0)]
            with patch.object(R,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                R.report(SimpleNamespace(output=output),cases)
            with (output/'report.csv').open() as stream:
                rows={r['kernel']:r for r in csv.DictReader(stream)}
            self.assertEqual(set(rows),{'aten_missing','aten_failed'})
            self.assertEqual(rows['aten_missing']['paper_raised_passed'],'')
            self.assertEqual(rows['aten_failed']['paper_raised_passed'],'False')
            self.assertEqual(rows['aten_failed']['paper_raised_native_agreement'],'True')
            self.assertEqual(rows['aten_failed']['raised_us'],'')
            self.assertEqual(rows['aten_failed']['timing_pairs'],'')

    def test_residual_loops_and_failed_abi_cannot_enter_timing_cohort(self):
        complete = dict(emitted=['library'], abi_ok=True, counts=dict(linalg=0, loops=0), rank=0)
        residual = dict(complete, counts=dict(linalg=0, loops=1))
        no_abi = dict(complete, abi_ok=False)
        self.assertEqual(R.complete_variants(dict(variants=[residual, no_abi, complete])), [complete])

    def test_followup_preserves_primary_and_only_selects_complete_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'aten').mkdir()
            (root/'aten/paper_cases.csv').write_text('kernel\naten_case\n')
            primary=root/'primary'; follow=root/'follow'
            for p in [primary,follow]: (p/'aten_case').mkdir(parents=True)
            incomplete=dict(variants=[dict(rank=0,emitted=['small'],abi_ok=True,
                                          counts=dict(linalg=1,loops=0))])
            complete=dict(variants=[dict(rank=2,emitted=['whole'],abi_ok=True,
                                        counts=dict(linalg=0,loops=0))])
            p=primary/'aten_case/result.json';p.write_text(json.dumps(incomplete))
            f=follow/'aten_case/result.json';f.write_text(json.dumps(complete))
            with patch.object(R,'ROOT',root):
                self.assertFalse(R.inventory(primary)[0]['complete_match'])
                selected=R.inventory(primary,[follow])[0]
                self.assertTrue(selected['complete_match'])
                self.assertEqual(selected['campaign'],str(follow))
                self.assertEqual(selected['rank'],2)
                f.write_text(json.dumps(incomplete))
                self.assertFalse(R.inventory(primary,[follow])[0]['complete_match'])
                p.unlink()
                with self.assertRaisesRegex(ValueError,'not finished'):
                    R.inventory(primary,[follow])


if __name__=='__main__': unittest.main()
