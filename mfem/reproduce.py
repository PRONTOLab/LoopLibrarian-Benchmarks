"""Build/run all 18 supported MFEM extractions and the integrated PCG fixture."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import gpu_target
from common import ROOT, OUT, command, environment, save, setting, sha
import local_execution

BASE=ROOT/'mfem/inputs'
CASES=json.loads((BASE/'manifest.json').read_text())
CASES.append(dict(kernel='pcg_application',function='mfem_app_ex9p_mass_convection_2d',
                  source='application_extractions/ex9p_mass_convection_2d.c',macro='MFEM_APP_EX9P'))

def cutensornet_root():
    configured = setting('cutensornet_root', os.environ.get('POLYGEIST_CUTENSORNET_ROOT'))
    if configured:
        return Path(configured)
    return next((p for p in (Path('/opt/cuquantum'), Path('/usr/local/cuquantum'))
                 if (p/'include/cutensornet.h').is_file()), None)

def build(row, contraction_library='cutensornet'):
    name=row['kernel'];d=OUT/name;d.mkdir(parents=True,exist_ok=True)
    pcg=name=='pcg_application'
    harness=BASE/('application_extractions/mfem_application_reference_harness.c' if pcg else 'benchmarks/mfem_full_output_validation.c')
    env=environment(POLYGEIST_BUILD_WORK_DIR=d/'work',POLYGEIST_ABI_BACKEND='cuda',
                    POLYGEIST_KEEP_DEPENDENCIES=1,POLYGEIST_MATCH_MAX_PARALLEL_DIMS=2,POLYGEIST_MATCH_MAX_OPERAND_RANK=2,
                    POLYGEIST_MATCH_DISABLED_KERNELS=('cublasDgemm cublasDgemv'
                        if contraction_library == 'cutensornet' else 'cutensornetNetworkContract'))
    if contraction_library == 'cutensornet':
        sdk = cutensornet_root()
        if not sdk or not (sdk/'include/cutensornet.h').is_file():
            raise RuntimeError('MFEM execution requires the cuTensorNet SDK; set cutensornet_root in config.json or POLYGEIST_CUTENSORNET_ROOT. Matching does not require the SDK.')
        env['POLYGEIST_CUTENSORNET_ROOT'] = str(sdk)
    args=[ROOT/'scripts/correctness/polygeist_build.sh','--target='+gpu_target.compiler_target(),'--function='+row['function'],
          '--harness='+str(harness),BASE/row['source'],'-D'+row['macro'],
          '-DMFEM_BENCH_NE='+('2' if pcg else '1024'),'-o',d/'program']
    if pcg:args+=['-DBENCH_ITERS=1','-DMFEM_REPLAY_WARMUPS=0','-DMFEM_PUBLICATION_WARMUPS=0',
                  '-DMFEM_PUBLICATION_SAMPLES=1','-DMFEM_DEBUG_OUTPUT_SUMMARY=1']
    command(args,d/'build.log',env)
    proof=json.loads((d/'work/match.json').read_text())
    save(d/'build.json',dict(case=name,source_sha256=sha(BASE/row['source']),harness_sha256=sha(harness),
         executable_sha256=sha(d/'program'),emitted=proof['emitted'],contraction_library=contraction_library))

def run(row):
    local_execution.require_gpu_target()
    name=row['kernel'];d=OUT/name;b=json.loads((d/'build.json').read_text())
    assert sha(d/'program')==b['executable_sha256']
    target='/tmp/looplibrarian-artifact-mfem/'+name
    assert local_execution.stage(d/'program',target)==0
    runtime_env=local_execution.runtime_env()
    sdk=cutensornet_root()
    if sdk:
        runtime_env+=' LD_LIBRARY_PATH='+shlex.quote(str(sdk/'lib')+':'+gpu_target.library_path())
    cmd='ulimit -s unlimited; chmod +x '+shlex.quote(target)+'; '+runtime_env+' POLYGEIST_V2_TRACE=1 timeout 300 '+shlex.quote(target)
    rc,log=local_execution.run(cmd,timeout=320);(d/'run.log').write_text(log)
    valid=rc==0 and ('correctness=PASS' in log or ('ALL CHECKS PASSED' in log))
    calls=re.findall(r'POLYGEIST_V2_CALL backend=cuda call=(\S+)',log)
    valid=valid and bool(calls)
    result=dict(case=name,run_rc=rc,correctness=valid,static_calls=len(b['emitted']),vendor_invocations=len(calls),command=cmd)
    save(d/'result.json',result)
    if not valid:raise RuntimeError(f'MFEM {name} failed; see {d}/run.log')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['match','report','build','run','summarize','all'],default='all')
    p.add_argument('--cases',nargs='+')
    p.add_argument('--contraction-library', choices=['cutensornet', 'cublas'], default='cutensornet',
                   help='library used for contraction matching; cuTensorNet is the paper backend')
    a=p.parse_args()
    available = list(CASES)
    if a.stage in ['match','report']:
        available += [dict(kernel=f'elasticity_qpoint_{dim}d',
                           function=f'mfem_elasticity_qpoint_{dim}d_scalarized',
                           source='normalized/elasticity_scalarized.c') for dim in (2,3)]
    rows=[r for r in available if not a.cases or r['kernel'] in a.cases]
    if not rows or (a.cases and set(a.cases)-{r['kernel'] for r in rows}):p.error('unknown MFEM case')
    if a.stage in ['match','report']:
        import matching
        report = matching.reproduce(rows, a.contraction_library) if a.stage == 'match' else matching.summarize(rows)
        print(json.dumps({k:report[k] for k in ['recognized_inputs','abi_lowered_inputs','current_contraction_regions','paper_contraction_sites']}))
        return
    if a.stage in ['build','all']:
        with ThreadPoolExecutor(max_workers=int(setting('jobs',2))) as pool:
            list(pool.map(lambda row: build(row, a.contraction_library), rows))
    if a.stage in ['run','all']:
        for row in rows:run(row)
    if a.stage in ['run','summarize','all']:
        results=[json.loads((OUT/r['kernel']/'result.json').read_text()) for r in rows]
        assert all(r['correctness'] for r in results)
        save(OUT/'summary.json',dict(results=results,passed=len(results),
             library_calls=sum(r['static_calls'] for r in results),
             historical_sites=128,qualification='Emitted contraction calls and published source sites are reported separately; no MFEM timing claim.'))
if __name__=='__main__':main()
