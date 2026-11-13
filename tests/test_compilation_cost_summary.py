"""Incomplete or failed builds must not become successful table entries."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'compilation_cost/reproduce.py'
spec=importlib.util.spec_from_file_location('compilation_reproduce',path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class CompilationSummary(unittest.TestCase):
    def test_summary_uses_recorded_cohort_without_current_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cases=[dict(suite='aten',kernel='archived',source='unavailable.c')]
            (root/'measurement.json').write_text(json.dumps(dict(cases=cases,repetitions=2)))
            for n in (1,2):
                dest=root/'aten/archived'/('full-'+str(n));dest.mkdir(parents=True)
                (dest/'result.json').write_text(json.dumps(dict(status='ok',wall_seconds=n,
                    repetition=n,returncode=0)))
            with patch('sys.argv',['reproduce.py','--stage','summarize','--output',str(root)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                driver.main()
            summary=json.loads((root/'summary.json').read_text())
            full=next(t for t in summary['tables'] if t['mode']=='full')
            self.assertEqual(full['requested_inputs'],1)
            self.assertEqual(full['complete_inputs'],1)
            self.assertEqual(full['mean'],1.5)

    def test_first_three_requires_repetitions_one_two_three(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cases=[dict(suite='aten',kernel=k) for k in ['first_three','missing_first']]
            for case,repetitions in zip(cases,[(1,2,3,4,5),(2,3,4)]):
                for n in repetitions:
                    dest=root/'aten'/case['kernel']/('full-'+str(n));dest.mkdir(parents=True)
                    # A later failure must not erase the valid three-repetition subset.
                    status='error' if n==5 else 'ok'
                    (dest/'result.json').write_text(json.dumps(dict(status=status,
                        wall_seconds=10*n,repetition=n,returncode=int(status!='ok'))))
            driver.summarize(root,cases,5)
            summary=json.loads((root/'summary.json').read_text())
            full=next(t for t in summary['tables'] if t['mode']=='full')
            first=next(t for t in summary['first_three_tables'] if t['mode']=='full')
            self.assertEqual(full['complete_inputs'],0)
            self.assertEqual(first['complete_inputs'],1)
            self.assertEqual(first['incomplete_inputs'],1)
            self.assertEqual(first['mean'],20)
            missing=next(r for r in summary['inputs']
                         if r['mode']=='full' and r['kernel']=='missing_first')
            self.assertIsNone(missing['first_three_median_seconds'])
            self.assertTrue((root/'first-three-compilation-table.csv').is_file())

    def test_metric_denominators_do_not_treat_missing_measurements_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cases=[dict(suite='mfem',kernel=k) for k in ['measured','missing']]
            for case in cases:
                for n,seconds in enumerate([4,6,8],1):
                    dest=root/'mfem'/case['kernel']/('full-'+str(n));dest.mkdir(parents=True)
                    row=dict(status='ok',wall_seconds=seconds,repetition=n,returncode=0)
                    if case['kernel']=='measured':
                        row.update(process_timing=dict(peak_rss_kib=100*n),
                            stages=[dict(label='[2/9] raising',duration_seconds=n)],
                            proof_measurements=dict(saturation_seconds=0.1*n,timeouts=0,worker_errors=0))
                    (dest/'result.json').write_text(json.dumps(row))
            driver.summarize(root,cases,3)
            summary=json.loads((root/'summary.json').read_text())
            rss=next(r for r in summary['metric_tables'] if r['mode']=='full' and r['metric']=='peak_rss_kib')
            self.assertEqual(rss['median'],200)
            self.assertEqual(rss['complete_inputs'],1)
            self.assertEqual(rss['incomplete_inputs'],1)
            self.assertTrue((root/'measurement-tables.csv').is_file())

    def test_failed_repetition_excludes_input_from_complete_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cases=[dict(suite='aten',kernel=k) for k in ['all_pass','one_failure','missing']]
            for case in cases[:2]:
                for n,seconds in enumerate([3,4,100],1):
                    dest=root/'aten'/case['kernel']/('full-'+str(n));dest.mkdir(parents=True)
                    status='error' if case['kernel']=='one_failure' and n==3 else 'ok'
                    (dest/'result.json').write_text(json.dumps(dict(status=status,wall_seconds=seconds,
                        repetition=n,returncode=0 if status=='ok' else 1)))
            tables=driver.summarize(root,cases,3)
            full=next(t for t in tables if t['mode']=='full')
            self.assertEqual(full['requested_inputs'],3)
            self.assertEqual(full['complete_inputs'],1)
            self.assertEqual(full['incomplete_inputs'],2)
            self.assertEqual(full['mean'],4)
            rows=json.loads((root/'summary.json').read_text())['inputs']
            failed=next(r for r in rows if r['kernel']=='one_failure' and r['mode']=='full')
            self.assertEqual(failed['status'],'incomplete')
            self.assertIsNone(failed['first_three_median_seconds'])
            self.assertEqual(len(failed['failures']),1)


if __name__=='__main__':unittest.main()
