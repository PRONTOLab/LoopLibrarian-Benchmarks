"""Cohort boundaries, resume selection, and matching/runtime routing."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('aten_cohort_driver', ROOT/'aten/reproduce.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class CohortDriver(unittest.TestCase):
    def test_inventory_is_exact_and_paper_is_a_fixed_subset(self):
        _, all_rows = driver.select_cases('all')
        _, paper = driver.select_cases('paper')
        self.assertEqual(len(all_rows), 598)
        self.assertEqual(len(paper), 115)
        self.assertEqual({r['kernel'] for r in all_rows}, {p.stem for p in (ROOT/'aten/inputs').glob('*.c')})
        self.assertTrue({r['kernel'] for r in paper} < {r['kernel'] for r in all_rows})
        for row in all_rows:
            self.assertEqual(hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest(), row['source_sha256'])
        with self.assertRaisesRegex(ValueError, 'outside'):
            driver.select_cases('paper', ['gather_cpu'])

    def test_all_models_are_explicit_even_when_reusing_paper_recipes(self):
        _, paper = driver.select_cases('paper')
        names = [r['kernel'] for r in paper] + ['aten_gather_cpu']
        strict = driver.all_matching_options(names, 'strict')
        self.assertTrue(all(r['match_options']['numeric_model'] == 'strict' for r in strict))
        self.assertTrue(strict[-1]['match_options']['split_pointwise_inputs'])
        self.assertEqual(strict[-1]['match_options']['ranks'], [1])

    def test_resume_keeps_subset_and_rejects_cohort_and_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = SimpleNamespace(cohort='all', cases=['gather_cpu'], numeric_model='strict', timing_correctness='strict')
            driver.record_selection(args, out)
            args.cases = None
            self.assertEqual(driver.record_selection(args, out)[1], ['aten_gather_cpu'])
            args.cohort = 'paper'
            with self.assertRaisesRegex(ValueError, 'another ATen cohort'):
                driver.record_selection(args, out)
            args.cohort = 'all'
            saved = json.loads((out/'selection.json').read_text())
            saved['source_sha256']['aten_gather_cpu'] = 'changed'
            (out/'selection.json').write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'source hashes'):
                driver.record_selection(args, out)

    def test_full_match_does_not_run_gpu_and_records_reusable_options(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver, 'command') as command:
            out = Path(tmp)
            # The top-level runner creates provenance before calling this driver.
            (out/'provenance.json').write_text('{}')
            driver.main(['--cohort','all','--stage','match','--cases','gather_cpu','--output',tmp])
            self.assertEqual(command.call_count, 1)
            cmd = list(map(str, command.call_args.args[0]))
            self.assertEqual(cmd[cmd.index('--stage')+1], 'match')
            self.assertEqual(cmd[cmd.index('--scope')+1], 'all')
            self.assertIn('--case-options', cmd)
            self.assertNotIn('--shape-specs', cmd)
            saved = json.loads((out/'selection.json').read_text())
            self.assertEqual(saved['cases'], ['aten_gather_cpu'])

    def test_build_resumes_without_rematching_and_rejects_new_case_selection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver, 'command') as command:
            out = Path(tmp)
            args = SimpleNamespace(cohort='all',cases=['gather_cpu'],numeric_model='strict',timing_correctness='strict')
            driver.record_selection(args, out)
            (out/'matching').mkdir()
            (out/'matching/provenance.json').write_text(json.dumps(dict(
                arguments={'shape_specs':None}, selected=['aten_gather_cpu'])))
            driver.main(['--cohort','all','--stage','build','--output',tmp])
            cmd = list(map(str, command.call_args.args[0]))
            self.assertEqual(command.call_count, 1)
            self.assertEqual(cmd[cmd.index('--stage')+1], 'build')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.main(['--cohort','all','--stage','run','--cases','embedding','--output',tmp])

    def test_paper_all_keeps_shapes_protocol_and_invokes_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            def command(cmd, *args, **kwargs):
                if 'report' in cmd:
                    (out/'gpu').mkdir(exist_ok=True)
                    (out/'gpu/report.json').write_text(json.dumps(dict(summary={'timing_pairs_validated':1})))
            with patch.object(driver, 'command', side_effect=command) as calls, contextlib.redirect_stdout(io.StringIO()):
                driver.main(['--cohort','paper','--cases','mm','--output',tmp,'--timing-correctness','publication'])
            commands = [list(map(str, c.args[0])) for c in calls.call_args_list]
            match = commands[0]
            self.assertIn(str(ROOT/'aten/paper_shapes.json'), match)
            self.assertIn(str(ROOT/'aten/matching_options.json'), match)
            self.assertEqual([c[c.index('--stage')+1] for c in commands if '--stage' in c], ['build','run','report'])
            self.assertTrue(commands[-1][1].endswith('plot_results.py'))
            for cmd in commands[1:4]:
                self.assertIn('--native-paper-inputs', cmd)
                self.assertEqual(cmd[cmd.index('--timing-correctness')+1], 'publication')

    def test_full_cohort_cannot_use_paper_plot_or_criterion(self):
        for extra in (['--stage','plot'], ['--timing-correctness','publication']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.main(['--cohort','all',*extra])

    def test_plot_restores_saved_policy_without_running_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = SimpleNamespace(cohort='paper',cases=['mm'],numeric_model='strict',timing_correctness='publication')
            driver.record_selection(args, out)
            (out/'matching').mkdir()
            shapes = ROOT/'aten/paper_shapes.json'
            (out/'matching/provenance.json').write_text(json.dumps(dict(
                arguments={'shape_specs':str(shapes)},selected=['aten_mm'],
                hashes={str(shapes):hashlib.sha256(shapes.read_bytes()).hexdigest()})))
            def command(cmd, *args, **kwargs):
                if 'report' in cmd:
                    (out/'gpu').mkdir(exist_ok=True)
                    (out/'gpu/report.json').write_text(json.dumps(dict(summary={'timing_pairs_validated':1})))
            with patch.object(driver, 'command', side_effect=command) as calls, contextlib.redirect_stdout(io.StringIO()):
                driver.main(['--cohort','paper','--stage','plot','--output',tmp])
            commands = [list(map(str, c.args[0])) for c in calls.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][commands[0].index('--stage')+1], 'report')
            self.assertEqual(commands[0][commands[0].index('--timing-correctness')+1], 'publication')
            self.assertTrue(commands[1][1].endswith('plot_results.py'))


if __name__ == '__main__':
    unittest.main()
