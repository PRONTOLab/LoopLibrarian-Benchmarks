import importlib.util
from pathlib import Path
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
    def counts(self, none_published=8, none_total=8):
        return dict(full=dict(published_cohort_recognized=17, recognized=19),
                    none=dict(published_cohort_recognized=none_published, recognized=none_total),
                    speq=dict(linearized_gemm_matches=1))

    def test_additional_recognitions_do_not_change_the_published_set(self):
        self.assertTrue(eqsat.submitted_counts_match(self.counts(none_total=10)))

    def test_changed_ablation_count_in_the_published_set_is_reported(self):
        self.assertFalse(eqsat.submitted_counts_match(self.counts(11, 12)))

    def test_matching_total_cannot_hide_a_missing_published_match(self):
        self.assertFalse(eqsat.submitted_counts_match(self.counts(7, 8)))


if __name__ == '__main__':
    unittest.main()
