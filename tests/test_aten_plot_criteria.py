"""A timing plot must preserve the criterion and input recipe it validated."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'aten'))
import plot_results
from plot_results import timing_group, validate_paper_cohort


class PlotCriteria(unittest.TestCase):
    def test_plot_rejects_full_inventory_and_duplicate_cases(self):
        validate_paper_cohort(dict(cases=[dict(kernel='aten_mm')]))
        for data in (dict(cohort='all',cases=[dict(kernel='aten_mm')]),
                     dict(cases=[dict(kernel='aten_gather_cpu')]),
                     dict(cases=[dict(kernel='aten_mm')]*2)):
            with self.assertRaises(ValueError):
                validate_paper_cohort(data)

    def result(self, recipe='paper', policy='strict', strict=False, publication=False):
        return dict(timing=dict(input_kind=recipe, correctness_policy=policy),
                    correctness={recipe: dict(passed=strict, publication_passed=publication)})

    def test_publication_pass_is_visibly_separate_from_strict_pass(self):
        self.assertEqual(timing_group(self.result(strict=True)), 'paper')
        self.assertEqual(timing_group(self.result(policy='publication', publication=True)),
                         'publication')
        with self.assertRaises(AssertionError):
            timing_group(self.result(policy='strict', publication=True))

    def test_alternative_recipe_remains_separate(self):
        self.assertEqual(timing_group(self.result(recipe='native-paper', strict=True)),
                         'alternative')
        self.assertEqual(timing_group(self.result(recipe='native-paper', policy='publication',
                                                 publication=True)), 'alternative')
        with self.assertRaises(AssertionError):
            timing_group(self.result(recipe='native-paper'))

    def test_failed_and_unknown_policies_cannot_enter_plot(self):
        for policy in ('strict', 'publication', 'unknown'):
            with self.assertRaises(AssertionError):
                timing_group(self.result(policy=policy))

    def test_plot_counts_only_valid_pairs_and_retains_unplotted_csv_rows(self):
        root = Path(__file__).resolve().parents[1]
        with (root/'aten/paper_cases.csv').open() as stream:
            cases = list(csv.DictReader(stream))
        published = dict(native_us=1000, raised_us=500, timing_protocol='minimum of 5 after 5 warmups')
        runtime = self.result(policy='publication', publication=True)
        runtime['binary_sha256'] = 'validated-binary'
        runtime['timing'].update(passed=True, minimum_ms={'raised': .5, 'native_aten_cuda': 1},
            samples=[dict(path=path, sample=i, ms=value+i*.1)
                     for path, value in [('raised', .5), ('native_aten_cuda', 1)] for i in range(5)])
        rows = [dict(kernel=row['kernel'], complete_match=False, published=published) for row in cases]
        rows[0].update(complete_match=True, build=dict(rc=0, binary_sha256='validated-binary'), runtime=runtime)
        rows[1].update(complete_match=True, build=dict(rc=1), runtime=dict(timing=dict(passed=False)))
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)/'report.json'
            report.write_text(json.dumps(dict(cases=rows, summary={'timing_pairs_validated':1})))
            output = Path(tmp)/'plots'
            with patch.object(plot_results.subprocess, 'run') as render, contextlib.redirect_stdout(io.StringIO()):
                plot_results.main(['--report',str(report),'--output',str(output)])
            self.assertEqual(render.call_count, 2)
            with (output/'aten-current-data.csv').open() as stream:
                plotted_rows = list(csv.DictReader(stream))
            self.assertEqual(len(plotted_rows), len(cases))
            self.assertEqual(plotted_rows[0]['status'], 'validated_original_inputs_publication')
            self.assertEqual(plotted_rows[1]['status'], 'build_failed')
            self.assertTrue(all(row['native_us']==row['raised_us']=='' for row in plotted_rows[1:]))
            summary = json.loads((output/'plot-summary.json').read_text())
            self.assertEqual(summary['timed'], 1)
            self.assertEqual(summary['paper_cohort'], len(cases))
            self.assertEqual(len(summary['unplotted']), len(cases)-1)
            self.assertIn(f'1 of {len(cases)} cases plotted', (output/'aten-current.tex').read_text())

            report.write_text(json.dumps(dict(cases=rows[:3], summary={'timing_pairs_validated':1})))
            with patch.object(plot_results.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()):
                plot_results.main(['--report',str(report),'--output',str(output)])
            self.assertIn(f'1 of {len(cases)} cases plotted (3 selected)',
                          (output/'aten-current.tex').read_text())


if __name__ == '__main__':
    unittest.main()
