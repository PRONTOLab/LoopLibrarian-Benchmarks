"""Test vendor semantics against unchanged C fixtures; no compiler matching."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/artifact'))
from common import COMPILER,OUT,command,environment,save,sha
import remote

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT/'pva-vendor-compatibility')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    env=environment()
    include=[Path(env['PVA_OPERATOR_INCLUDE']),Path(env['NVCV_INCLUDE']),
             Path(env['CUPVA_SDK_ROOT'])/'include',COMPILER/'runtime']
    cuda=Path(env.get('CUDA_CROSS','/usr/local/cuda-12.6/targets/sbsa-linux'))
    include.append(cuda/'include')
    libs=Path(env['PVA_LIB_STAGE'])
    sources=[ROOT/'pva/vendor_compatibility.c',ROOT/'pva/inputs/image_processing.c',
             COMPILER/'runtime/polygeist_pva_image_rt.c']
    command([env.get('AARCH64_CC','aarch64-linux-gnu-gcc'),'-O2','-fno-fast-math',
        '-ffp-contract=off','-D_Float16=__fp16',*['-I'+str(x) for x in include],
        *sources,'-L'+str(libs),'-L'+str(cuda/'lib'),'-lpva_operator',
        '-lnvcv_types','-lcupva_host','-lcudart','-lm','-lpthread','-ldl',
        '-Wl,--allow-shlib-undefined','-o',out/'program'],out/'build.log',env)
    save(out/'build.json',dict(source_hashes={str(s):sha(s) for s in sources},
        executable_sha256=sha(out/'program'),scope=__doc__))
    target='/tmp/looplibrarian-artifact-pva/vendor-compatibility'
    if remote.upload(out/'program',target):raise RuntimeError('upload failed')
    rc,log=remote.run('chmod +x '+shlex.quote(target)+'; '+remote.runtime_env()+
        ' timeout 180 '+shlex.quote(target),timeout=200)
    (out/'run.log').write_text(log)
    rows=[]
    for line in log.splitlines():
        if not line.startswith('COMPARE,'):continue
        _,family,recipe,*fields=line.split(',')
        rows.append(dict(family=family,recipe=recipe,
            **{k:int(v) for k,v in (x.split('=') for x in fields)}))
    save(out/'summary.json',dict(returncode=rc,comparisons=rows,
        scope=__doc__,timing_claim=False,matching_claim=False))
    print(json.dumps(rows),flush=True)
    if rc or len(rows)!=15:raise SystemExit(1)

if __name__=='__main__':main()
