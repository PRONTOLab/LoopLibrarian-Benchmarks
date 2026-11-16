import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('coverage_table', ROOT/'aten/coverage_table.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class CoverageTable(unittest.TestCase):
    def test_inventory_counts_and_unevaluated_cases(self):
        report = coverage.reconcile(ROOT)
        summary = report['summary']
        self.assertEqual(summary['current']['all_kernels'], 598)
        self.assertEqual(summary['current']['native_library_kernels'], 52)
        self.assertEqual(summary['current']['matched'], 0)
        self.assertEqual(summary['templated_native'], 26)
        self.assertEqual(summary['matched_templated'], [])
        self.assertEqual(summary['not_evaluated'], 598)
        self.assertEqual(summary['incomplete_matches'], 0)
        self.assertFalse(summary['current_full_inventory'])

    def test_partial_and_failed_abi_are_not_complete_matches(self):
        self.assertFalse(coverage.complete(dict(emitted=['copy'], abi_ok=True,
            counts=dict(linalg=1, loops=0))))
        self.assertFalse(coverage.complete(dict(emitted=['gemm'], abi_ok=False,
            counts=dict(linalg=0, loops=0))))
        self.assertFalse(coverage.complete(dict(emitted=[], abi_ok=True,
            counts=dict(linalg=0, loops=0))))

    def test_partial_run_is_explicit_and_changed_source_is_rejected(self):
        inventory = coverage.unique_rows(ROOT/'aten/all_cases.csv')
        name, original = next(iter(inventory.items()))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'provenance.json').write_text(json.dumps(dict(selected=[name], arguments={})))
            dest = out/name
            dest.mkdir()
            result = dict(source_sha256=original['source_sha256'], status='recognized',
                variants=[dict(emitted=['example'], abi_ok=True, counts=dict(linalg=0, loops=0))])
            (dest/'result.json').write_text(json.dumps(result))
            report = coverage.reconcile(ROOT, out)
            self.assertEqual(report['summary']['current_attempted'], 1)
            self.assertEqual(report['summary']['current']['matched'], 1)
            self.assertFalse(report['summary']['current_full_inventory'])
            self.assertEqual(report['summary']['incomplete_matches'], 0)
            self.assertEqual(report['summary']['not_evaluated'], 597)
            result['source_sha256'] = 'changed'
            (dest/'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'Source differs'):
                coverage.reconcile(ROOT, out)

    def test_duplicate_native_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'audit.csv'
            path.write_text('kernel,status\nexample,yes\nexample,no\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate kernel'):
                coverage.unique_rows(path)


if __name__ == '__main__':
    unittest.main()
