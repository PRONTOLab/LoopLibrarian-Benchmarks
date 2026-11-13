"""Benchmark inputs must work in a source-only package without retained runs."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArtifactLayout(unittest.TestCase):
    def test_real_source_directories(self):
        for name in ['aten/inputs', 'aten/harnesses', 'aten/metadata',
                     'mfem/inputs', 'llama/inputs', 'polybench/sources']:
            path = ROOT / name
            self.assertTrue(path.is_dir(), name)
            self.assertFalse(path.is_symlink(), name)
            self.assertFalse(any(p.is_symlink() for p in path.rglob('*')), name)
        self.assertEqual(len(list((ROOT / 'aten/inputs').glob('aten_*.c'))), 598)

    def test_inputs_resolve_without_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            for name in ['aten/inputs', 'mfem/inputs', 'llama/inputs']:
                shutil.copytree(ROOT / name, package / name)
            scripts = package / 'scripts/correctness'
            scripts.mkdir(parents=True)
            shutil.copyfile(ROOT / 'scripts/correctness/ll_paths.py', scripts / 'll_paths.py')
            spec = importlib.util.spec_from_file_location('packaged_paths', scripts / 'll_paths.py')
            paths = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(paths)
            self.assertFalse((package / 'results/archive').exists())
            with (ROOT / 'aten/paper_cases.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            shapes = json.loads((ROOT / 'aten/paper_shapes.json').read_text())
            self.assertEqual(len(rows), 115)
            self.assertEqual({r['kernel'] for r in rows}, {r['kernel'] for r in shapes})
            for row in rows:
                source = paths.source_path(row['source'])
                self.assertTrue(source.is_file(), row['source'])
                self.assertEqual(source.parent, package / 'aten/inputs')
                for prefix in ['issues/', 'results/archive/issues/',
                               '/home/arjaiswal/Polygeist/issues/']:
                    self.assertEqual(paths.source_path(prefix + 'aten_c_kernels/' + source.name),
                                     source)
            mfem = package / 'mfem/inputs'
            rows = json.loads((mfem / 'manifest.json').read_text())
            self.assertEqual(len(rows), 18)
            for row in rows:
                self.assertTrue((mfem / row['source']).is_file(), row['source'])
            for name in ['benchmarks/mfem_full_output_validation.c',
                         'application_extractions/ex9p_mass_convection_2d.c',
                         'application_extractions/mfem_application_reference_harness.c']:
                self.assertTrue((mfem / name).is_file(), name)
            self.assertTrue((package / 'llama/inputs/llama2_extended_forward_bench.c').is_file())


if __name__ == '__main__':
    unittest.main()
