import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
common = types.ModuleType('common')
common.ROOT = common.OUT = common.INPUTS = ROOT
common.save = lambda *args: None
coverage = types.ModuleType('coverage')
coverage.campaign = lambda *args, **kwargs: None
spec = importlib.util.spec_from_file_location('eqsat_summary', ROOT/'eqsat/reproduce.py')
eqsat = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'common': common, 'coverage': coverage,
                               'speq': types.ModuleType('speq')}):
    spec.loader.exec_module(eqsat)


class EqSatSummaryTest(unittest.TestCase):
    def counts(self, full=17, none=8):
        return dict(full=dict(recognized=full), none=dict(recognized=none),
                    speq=dict(linearized_gemm_matches=1))

    def test_published_counts_use_the_complete_inventory(self):
        self.assertTrue(eqsat.submitted_counts_match(self.counts()))

    def test_corrected_counts_are_different_from_the_published_counts(self):
        self.assertFalse(eqsat.submitted_counts_match(self.counts(full=19, none=11)))

    def test_additional_full_recognitions_are_included_in_comparison(self):
        self.assertFalse(eqsat.submitted_counts_match(self.counts(full=19)))

    def summarize(self, selected, full, none):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = [dict(name=f'program_{i}') for i in range(selected)]
            for arm, recognized in [('full', full), ('none', none)]:
                for i, row in enumerate(rows):
                    path = out/'ll'/row['name']/arm/'result.json'
                    path.parent.mkdir(parents=True)
                    lowering_failure = arm == 'none' and i == 0
                    path.write_text(json.dumps(dict(name=row['name'], ablation=arm,
                        recognized=i < recognized, abi_ok=i < recognized and not lowering_failure,
                        error='lowering failed' if lowering_failure else None)))
            for name, results in [('speq-native', [dict(n_match=0)] * selected),
                                  ('speq-linearized', [dict(n_match=1)])]:
                (out/name).mkdir()
                (out/name/'results.json').write_text(json.dumps(results))
            with patch.object(eqsat, 'OUT', out), patch.object(eqsat, 'save') as save, \
                    contextlib.redirect_stdout(io.StringIO()):
                eqsat.summarize(rows)
            return save.call_args.args[1]

    def test_corrected_comparison_reports_successfully_and_preserves_lowering_failures(self):
        report = self.summarize(selected=30, full=19, none=11)
        self.assertTrue(report['full_cohort'])
        self.assertEqual(report['selected_inputs'], 30)
        self.assertFalse(report['submitted_counts_reproduced'])
        for arm, count in [('full', 19), ('none', 11)]:
            self.assertEqual(report['totals'][arm]['inputs'], 30)
            self.assertEqual(report['totals'][arm]['recognized'], count)
            self.assertEqual(len(report['totals'][arm]['recognized_names']), count)
            self.assertNotIn('published_cohort_recognized', report['totals'][arm])
        self.assertEqual(report['totals']['none']['abi_lowered'], 10)
        self.assertEqual(report['totals']['none']['errors'],
                         [dict(name='program_0', error='lowering failed')])

    def test_partial_inventory_does_not_claim_published_counts_reproduced(self):
        report = self.summarize(selected=17, full=17, none=8)
        self.assertFalse(report['full_cohort'])
        self.assertFalse(report['submitted_counts_reproduced'])


if __name__ == '__main__':
    unittest.main()
