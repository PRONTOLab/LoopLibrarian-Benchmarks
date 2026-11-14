"""A timing plot must preserve the criterion and input recipe it validated."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'aten'))
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


if __name__ == '__main__':
    unittest.main()
