"""Diagnose reduction-order differences using the same C body in FP64.

This relinks recorded objects for diagnosis only. It never replaces a fresh
paper correctness result, changes tolerances, or supplies timing measurements.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/artifact'))
from common import command,environment,save,sha
import remote

CASES=['aten_sparse_norm_cpu','aten_conv_transpose3d_grad_weight_cpu',
       'aten_slow_conv3d_backward_weight_cpu']

def build(name,base,out):
    src=base/name;dest=out/name;dest.mkdir(parents=True,exist_ok=True)
    record=json.loads((src/'build.json').read_text())
    if record['rc']!=0:raise ValueError('successful recorded build required: '+name)
    code=(src/(name+'.c')).read_text()
    (dest/'fp64.c').write_text('#define '+name+' fp64_reference\n#define float double\n#define sqrtf sqrt\n'+code)
    preamble=(src/'harness.c').read_text().split('/* Included after the fixture-specific declarations')[0]
    if '#define NS 0' not in preamble:raise ValueError('diagnostic requires no scalar parameters')
    arity=2 if name=='aten_sparse_norm_cpu' else 3
    fp64_call='fp64_reference('+','.join('wide['+str(i)+']' for i in range(arity))+');'
    body=r'''
int main(void) {
  float *original[NA],*serial[NA],*device[NA],*gpu[NA];double *wide[NA];
  for(int a=0;a<NA;++a) {
    original[a]=malloc(sizes[a]*sizeof(float));serial[a]=malloc(sizes[a]*sizeof(float));
    gpu[a]=malloc(sizes[a]*sizeof(float));wide[a]=malloc(sizes[a]*sizeof(double));
    if(!original[a]||!serial[a]||!gpu[a]||!wide[a])return 2;
    if(cudaMalloc((void**)&device[a],sizes[a]*sizeof(float))!=cudaSuccess)return 2;
    for(size_t i=0;i<sizes[a];++i) {
      original[a][i]=(float)paper_value(a,i);serial[a][i]=original[a][i];
      wide[a][i]=(double)original[a][i];
    }
  }
  invoke_reference(serial);
  FP64_CALL
  for(int path=0;path<2;++path) {
    for(int a=0;a<NA;++a)
      if(cudaMemcpy(device[a],original[a],sizes[a]*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)return 2;
    if(path) {invoke_native(device);complete_native(device);}else invoke(device);
    if(cudaDeviceSynchronize()!=cudaSuccess)return 2;
    size_t count=0,bad=0,serialbad=0;double maximum=0,serialmaximum=0;
    for(int a=0;a<NA;++a) if(outputs[a]) {
      if(cudaMemcpy(gpu[a],device[a],sizes[a]*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;
      for(size_t i=0;i<sizes[a];++i) {
        double reference=wide[a][i],error=fabs((double)gpu[a][i]-reference);
        double serialerror=fabs((double)serial[a][i]-reference);
        double bound=ABS_TOL+REL_TOL*fabs(reference);
        bad+=!isfinite(error)||error>bound;serialbad+=!isfinite(serialerror)||serialerror>bound;
        if(error>maximum)maximum=error;if(serialerror>serialmaximum)serialmaximum=serialerror;++count;
      }
    }
    printf("FP64_CHECK,path=%s,count=%zu,bad=%zu,serial_bad=%zu,maxabs=%.17g,serial_maxabs=%.17g\n",
      path?"native_aten_cuda":"raised",count,bad,serialbad,maximum,serialmaximum);
  }
  for(int a=0;a<NA;++a) {free(original[a]);free(serial[a]);free(gpu[a]);free(wide[a]);cudaFree(device[a]);}
  return 0;
}
'''.replace('FP64_CALL',fp64_call)
    declaration='void fp64_reference('+','.join(['double*']*arity)+');\n'
    (dest/'diagnostic.c').write_text(preamble+declaration+body)
    env=environment();cuda=Path(env.get('ATEN_NATIVE_CUDA_ROOT','/usr/local/cuda-12.6/targets/sbsa-linux'))
    cc=env.get('AARCH64_CC','aarch64-linux-gnu-gcc')
    for source,obj in [('fp64.c','fp64.o'),('diagnostic.c','diagnostic.o')]:
        command([cc,'-O2','-fno-fast-math','-ffp-contract=off','-I'+str(cuda/'include'),
                 '-c',dest/source,'-o',dest/obj],dest/'build.log',env)
    link=list(record['commands'][-1]);found=False
    for i,arg in enumerate(link):
        if arg==str(src/'work/harness.o'):link[i]=str(dest/'diagnostic.o');found=True
    if not found:raise ValueError('recorded harness object not found')
    link.insert(1,str(dest/'fp64.o'));link[link.index('-o')+1]=str(dest/'program')
    command(link,dest/'build.log',env)
    save(dest/'build.json',dict(scope=__doc__,base_build_sha256=sha(src/'build.json'),
        source_sha256=sha(src/(name+'.c')),executable_sha256=sha(dest/'program'),
        tolerance='Unchanged ABS_TOL/REL_TOL, evaluated against FP64 reference',
        object_hashes={arg:sha(Path(arg)) for arg in link if arg.endswith('.o')}))
    return dest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[]
    for name in CASES:
        dest=build(name,args.gpu_results.resolve(),args.output.resolve())
        target='/tmp/looplibrarian-aten-numerics/'+name
        if remote.upload(dest/'program',target):raise RuntimeError('upload failed')
        rc,log=remote.run('chmod +x '+shlex.quote(target)+'; '+remote.runtime_env()+
            ' POLYGEIST_CUDNN_PLAN_CACHE=1 POLYGEIST_GPU_WORKSPACE_CACHE=1 timeout 240 '+
            shlex.quote(target),timeout=260)
        (dest/'run.log').write_text(log)
        lines=[line for line in log.splitlines() if line.startswith('FP64_CHECK,')]
        row=dict(case=name,returncode=rc,checks=lines,scope=__doc__)
        save(dest/'result.json',row);rows.append(row)
        print(name,rc,*lines,sep='\n',flush=True)
        if rc or len(lines)!=2:raise SystemExit(1)
    save(args.output/'summary.json',rows)

if __name__=='__main__':main()
