"""Additional recognized operations do not bypass the complete-output check."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LlamaResults(unittest.TestCase):
    def test_additional_coverage_still_requires_correct_finite_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            names = ['llama_native', 'llama_resident', 'llama_ggml']
            logits = ''.join(f'LOGIT,{i},0\n' for i in range(32000))
            for name in names:
                (output / (name + '.log')).write_text(logits)
            (output / 'runs.json').write_text(json.dumps([
                dict(name=name, returncode=0, samples_ms=[1.0] * 5,
                     cuda_graph_replay=False,
                     **({'wall_samples_ms': [1.2, 1.1, 1.3, 1.2, 1.4]}
                        if name == 'llama_resident' else {})) for name in names]))

            def summarize(recognized):
                (output / 'coverage.json').write_text(json.dumps([
                    dict(recognized=i < recognized, abi_ok=i < recognized)
                    for i in range(19)]))
                env = dict(os.environ, LL_ARTIFACT_OUTPUT=tmp,
                           PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
                               ['scripts/artifact', 'scripts/correctness']))
                return subprocess.run([sys.executable, str(ROOT / 'llama/reproduce.py'),
                                       '--stage', 'summarize'], env=env,
                                      capture_output=True, text=True)

            result = summarize(15)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertTrue(summary['expert_comparison_passed'])
            self.assertTrue(summary['coverage_count_meets_paper'])
            self.assertFalse(summary['coverage_matches_paper'])
            resident = next(r for r in summary['timings'] if r['name'] == 'llama_resident')
            self.assertEqual(resident['minimum_ms'], 1.0)
            self.assertEqual(resident['wall_minimum_ms'], 1.1)
            self.assertEqual(resident['wall_median_ms'], 1.2)
            self.assertIn('llama_ggml: minimum', result.stdout)
            self.assertNotEqual(summarize(13).returncode, 0)
            timings = (output / 'runs.json').read_text()
            rows = json.loads(timings)
            (output / 'runs.json').write_text(json.dumps(rows[:-1]))
            result = summarize(15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Expected one timing run', result.stderr)
            (output / 'runs.json').write_text(timings)
            (output / 'llama_resident.log').write_text(logits.replace('LOGIT,0,0\n', 'LOGIT,0,nan\n', 1))
            result = summarize(15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('non-finite logit 0', result.stderr)


if __name__ == '__main__':
    unittest.main()
