"""Partial replacements and nonexecutable proofs cannot inflate coverage."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/correctness'))
from aten_paper115_campaign import complete_variant


class MatchCoverage(unittest.TestCase):
    def test_computation_and_abi_must_both_be_complete(self):
        full = dict(emitted=['imported_spec'],abi_ok=True,counts=dict(linalg=0,loops=0))
        self.assertTrue(complete_variant(full))
        for change in [dict(abi_ok=False),dict(emitted=[]),dict(counts={}),
                       dict(counts=dict(linalg=1,loops=0)),dict(counts=dict(linalg=0,loops=1))]:
            self.assertFalse(complete_variant(dict(full,**change)))


if __name__=='__main__':
    unittest.main()
