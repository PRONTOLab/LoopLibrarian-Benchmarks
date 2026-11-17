"""The two ATen experiments keep their cohorts, execution needs and saved format."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('artifact_runner', ROOT/'reproduce.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
import gpu_target


class ATenExperiments(unittest.TestCase):
    def invoke(self, experiment, *options):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            out = Path(tmp)/'results'
            with patch.object(runner, 'provenance'), \
                    patch.object(runner, 'environment', side_effect=lambda **extra: dict(
                        POLYGEIST_DEPENDENCY_ROOT=str(ROOT/'third_party'),
                        **{key: str(value) for key, value in extra.items()})), \
                    patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run, \
                    patch.object(gpu_target, 'require_execution') as execution, \
                    patch.object(gpu_target, 'validate_toolkit') as toolkit:
                rc = runner.main([experiment, '--output', str(out), *options])
                self.assertEqual(rc, 0)
                command = list(map(str, run.call_args.args[0]))
                environment = run.call_args.kwargs['env']
                return command, environment, execution.call_count, toolkit.call_count

    def test_table5_defaults_to_matching_without_cuda_or_execution(self):
        for options in ([], ['--stage', 'all'], ['--stage', 'report'], ['--stage', 'summarize']):
            with self.subTest(options=options):
                command, env, execution, toolkit = self.invoke('aten_table5', *options)
                self.assertEqual(command[command.index('--cohort')+1], 'all')
                self.assertIn(command[command.index('--stage')+1], ('match', 'report', 'summarize'))
                self.assertEqual((execution, toolkit), (0, 0))
                self.assertEqual(env['LL_ARTIFACT_SUITE'], 'aten')

    def test_figure5_defaults_to_paper_pipeline_and_preserves_bundle_suite(self):
        command, env, execution, toolkit = self.invoke('aten_figure5')
        self.assertEqual(command[command.index('--cohort')+1], 'paper')
        self.assertEqual(command[command.index('--stage')+1], 'all')
        self.assertEqual((execution, toolkit), (1, 1))
        self.assertEqual(env['LL_ARTIFACT_SUITE'], 'aten')

    def test_figure5_saved_plot_and_report_do_not_need_cuda(self):
        for stage in ('report', 'plot'):
            with self.subTest(stage=stage):
                command, _, execution, toolkit = self.invoke('aten_figure5', '--stage', stage)
                self.assertEqual(command[command.index('--stage')+1], stage)
                self.assertEqual((execution, toolkit), (0, 0))

    def test_fixed_cohorts_and_matching_only_scope_cannot_be_overridden(self):
        for args in (['aten_table5', '--cohort', 'paper'],
                     ['aten_figure5', '--cohort', 'all'],
                     *(['aten_table5', '--stage', stage] for stage in ('build', 'run', 'plot'))):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                runner.main(args)

    def test_direct_scripts_and_main_runner_plan_the_same_experiments(self):
        for name, cohort, stage in [('table5', 'all', 'match'), ('figure5', 'paper', 'all')]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/'not-created'
                direct = subprocess.check_output([sys.executable, str(ROOT/'aten'/f'{name}.py'),
                    '--stage', 'plan', '--output', str(out)], text=True)
                routed = subprocess.check_output([sys.executable, str(ROOT/'reproduce.py'),
                    'aten_'+name, '--stage', 'plan', '--output', str(out)], text=True)
                self.assertEqual(direct, routed)
                command = shlex.split(direct.splitlines()[0])
                self.assertEqual(command[command.index('--cohort')+1], cohort)
                self.assertEqual(command[command.index('--stage')+1], stage)
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
