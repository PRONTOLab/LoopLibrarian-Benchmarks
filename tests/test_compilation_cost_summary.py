"""Incomplete or failed builds must not become successful table entries."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'compilation_cost/reproduce.py'
spec=importlib.util.spec_from_file_location('compilation_reproduce',path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class CompilationSummary(unittest.TestCase):
    def test_build_schedule_and_summary_follow_selected_modes(self):
        case=next(c for c in json.loads(path.with_name('inputs.json').read_text())['cases']
                  if c['suite']=='aten' and c['kernel']=='aten_avg_pool2d')
        for options,repetitions,modes in [([],3,['full']),
                (['--repetitions','2','--modes','full','none'],2,['full','none'])]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);compiler=root/'compiler';output=root/'results'
                for relative in ['build/bin/cgeist','build/bin/polygeist-opt',
                        'generic_solver/kernel_library_v3.mlir','generic_solver/lib_v3/import_manifest.json']:
                    f=compiler/relative;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fixture')
                source=root/case['source'];source.parent.mkdir(parents=True);source.write_text('fixture')
                (root/'aten/paper_shapes.json').write_text('{}')
                validator=SimpleNamespace(validate=lambda _: ({},None))
                with patch.object(driver,'ROOT',root), patch.object(driver,'COMPILER',compiler), \
                        patch.object(driver,'environment',return_value={}), \
                        patch.object(driver,'provenance'), patch.object(driver,'measure') as measure, \
                        patch.dict('sys.modules',spec_provenance=validator), \
                        patch.object(driver.sys,'path',driver.sys.path.copy()), \
                        patch('sys.argv',['reproduce.py','--output',str(output),
                              '--cases','aten/aten_avg_pool2d',*options]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    driver.main()
                calls=[(c.args[1],c.args[2]) for c in measure.call_args_list]
                self.assertCountEqual(calls,[(mode,n) for mode in modes
                                            for n in range(1,repetitions+1)])
                recorded=json.loads((output/'measurement.json').read_text())
                self.assertEqual(recorded['modes'],modes)
                self.assertEqual(recorded['repetitions'],repetitions)
                self.assertEqual(recorded['measurement_boundary'],driver.MEASUREMENT_BOUNDARY)
                with patch('sys.argv',['reproduce.py','--stage','summarize','--output',str(output)]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    driver.main()
                summary=json.loads((output/'summary.json').read_text())
                self.assertEqual([t['mode'] for t in summary['tables']],modes)
                self.assertEqual(summary['measurement_boundary'],driver.MEASUREMENT_BOUNDARY)
                self.assertEqual([r['requested'] for r in summary['inputs']],
                                 [repetitions]*len(modes))

    def test_summary_uses_recorded_cohort_without_current_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cases=[dict(suite='aten',kernel='archived',source='unavailable.c')]
            (root/'measurement.json').write_text(json.dumps(dict(cases=cases,repetitions=2)))
            for n in (1,2):
                dest=root/'aten/archived'/('full-'+str(n));dest.mkdir(parents=True)
                (dest/'result.json').write_text(json.dumps(dict(status='ok',wall_seconds=n,
                    repetition=n,returncode=0)))
            with patch('sys.argv',['reproduce.py','--stage','summarize','--output',str(root)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                driver.main()
            summary=json.loads((root/'summary.json').read_text())
            full=next(t for t in summary['tables'] if t['mode']=='full')
            self.assertEqual(full['requested_inputs'],1)
            self.assertEqual(full['complete_inputs'],1)
            self.assertEqual(full['mean'],1.5)
            self.assertEqual([t['mode'] for t in summary['tables']],['full','none'])
            self.assertEqual(summary['measurement_boundary'],driver.LEGACY_BOUNDARY)

    def test_different_measurement_boundaries_cannot_be_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cases=[dict(suite='aten',kernel='mixed')]
            for n,boundary in enumerate([driver.LEGACY_BOUNDARY,driver.MEASUREMENT_BOUNDARY],1):
                dest=root/'aten/mixed'/('full-'+str(n));dest.mkdir(parents=True)
                (dest/'result.json').write_text(json.dumps(dict(status='ok',wall_seconds=10,
                    repetition=n,returncode=0,measurement_boundary=boundary)))
            with self.assertRaisesRegex(ValueError,'measurement boundaries'):
                driver.summarize(root,cases,2,measurement_boundary=driver.MEASUREMENT_BOUNDARY)

    def test_matching_completion_does_not_require_a_linked_executable(self):
        for rc,proof_available,expected in [(0,True,'ok'),(1,True,'error'),(0,False,'error')]:
            with self.subTest(rc=rc,proof_available=proof_available), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'kernel.c').write_text('void kernel(void) {}')
                output=root/'out';dest=output/'polybench/kernel/full-1';work=dest/'work'
                case=dict(suite='polybench',kernel='kernel',function='kernel',source='kernel.c',cflags=[])

                def process(*args,**kwargs):
                    work.mkdir(parents=True)
                    if proof_available:
                        (work/'matched.mlir').write_text('module {}')
                        (work/'match.json').write_text(json.dumps(dict(emitted=[],elapsed_ms=1)))
                    (dest/'matcher.jsonl').write_text(json.dumps(dict(phase='linalg_match'))+'\n')
                    return SimpleNamespace(stdout=io.StringIO(
                        '[1/3] frontend\n[2/3] raising\n[3/3] equality saturation\n'),wait=lambda:rc)

                args=SimpleNamespace(output=output,compiler_state={},timeout=300,spec_cache=None)
                with patch.object(driver,'ROOT',root), patch.object(driver.subprocess,'Popen',side_effect=process), \
                        patch.object(driver.subprocess,'check_output',return_value='test load'), \
                        patch.object(driver,'environment',side_effect=lambda **kwargs:kwargs), \
                        contextlib.redirect_stdout(io.StringIO()):
                    driver.measure(case,'full',1,args)
                result=json.loads((dest/'result.json').read_text())
                self.assertEqual(result['status'],expected)
                self.assertEqual(result['measurement_boundary'],driver.MEASUREMENT_BOUNDARY)
                self.assertNotIn('executable_sha256',result)
                self.assertFalse((dest/'program').exists())
                if expected=='ok':
                    self.assertIsNotNone(result['matched_ir_sha256'])
                    self.assertIn('matching_stage_seconds',driver.measured_fields(result))

    def test_first_three_requires_repetitions_one_two_three(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cases=[dict(suite='aten',kernel=k) for k in ['first_three','missing_first']]
            for case,repetitions in zip(cases,[(1,2,3,4,5),(2,3,4)]):
                for n in repetitions:
                    dest=root/'aten'/case['kernel']/('full-'+str(n));dest.mkdir(parents=True)
                    # A later failure must not erase the valid three-repetition subset.
                    status='error' if n==5 else 'ok'
                    (dest/'result.json').write_text(json.dumps(dict(status=status,
                        wall_seconds=10*n,repetition=n,returncode=int(status!='ok'))))
            driver.summarize(root,cases,5)
            summary=json.loads((root/'summary.json').read_text())
            full=next(t for t in summary['tables'] if t['mode']=='full')
            first=next(t for t in summary['first_three_tables'] if t['mode']=='full')
            self.assertEqual(full['complete_inputs'],0)
            self.assertEqual(first['complete_inputs'],1)
            self.assertEqual(first['incomplete_inputs'],1)
            self.assertEqual(first['mean'],20)
            missing=next(r for r in summary['inputs']
                         if r['mode']=='full' and r['kernel']=='missing_first')
            self.assertIsNone(missing['first_three_median_seconds'])
            self.assertTrue((root/'first-three-compilation-table.csv').is_file())

    def test_metric_denominators_do_not_treat_missing_measurements_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cases=[dict(suite='mfem',kernel=k) for k in ['measured','missing']]
            for case in cases:
                for n,seconds in enumerate([4,6,8],1):
                    dest=root/'mfem'/case['kernel']/('full-'+str(n));dest.mkdir(parents=True)
                    row=dict(status='ok',wall_seconds=seconds,repetition=n,returncode=0)
                    if case['kernel']=='measured':
                        row.update(process_timing=dict(peak_rss_kib=100*n),
                            stages=[dict(label='[2/9] raising',duration_seconds=n)],
                            proof_measurements=dict(saturation_seconds=0.1*n,timeouts=0,worker_errors=0))
                    (dest/'result.json').write_text(json.dumps(row))
            driver.summarize(root,cases,3)
            summary=json.loads((root/'summary.json').read_text())
            rss=next(r for r in summary['metric_tables'] if r['mode']=='full' and r['metric']=='peak_rss_kib')
            self.assertEqual(rss['median'],200)
            self.assertEqual(rss['complete_inputs'],1)
            self.assertEqual(rss['incomplete_inputs'],1)
            self.assertTrue((root/'measurement-tables.csv').is_file())

    def test_failed_repetition_excludes_input_from_complete_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cases=[dict(suite='aten',kernel=k) for k in ['all_pass','one_failure','missing']]
            for case in cases[:2]:
                for n,seconds in enumerate([3,4,100],1):
                    dest=root/'aten'/case['kernel']/('full-'+str(n));dest.mkdir(parents=True)
                    status='error' if case['kernel']=='one_failure' and n==3 else 'ok'
                    (dest/'result.json').write_text(json.dumps(dict(status=status,wall_seconds=seconds,
                        repetition=n,returncode=0 if status=='ok' else 1)))
            tables=driver.summarize(root,cases,3)
            full=next(t for t in tables if t['mode']=='full')
            self.assertEqual(full['requested_inputs'],3)
            self.assertEqual(full['complete_inputs'],1)
            self.assertEqual(full['incomplete_inputs'],2)
            self.assertEqual(full['mean'],4)
            rows=json.loads((root/'summary.json').read_text())['inputs']
            failed=next(r for r in rows if r['kernel']=='one_failure' and r['mode']=='full')
            self.assertEqual(failed['status'],'incomplete')
            self.assertIsNone(failed['first_three_median_seconds'])
            self.assertEqual(len(failed['failures']),1)


if __name__=='__main__':unittest.main()
