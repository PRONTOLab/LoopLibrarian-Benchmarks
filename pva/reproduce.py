"""Raise public image fixtures, prove imported references and execute on PVA."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/artifact'),str(ROOT/'scripts/correctness')]
from common import COMPILER, OUT, command, environment, save, sha
import remote

CASES=[dict(name='morphology_'+dtype,dtype=dtype,ctype=ctype,minimum=minimum,
            family='Morphology',vendor='pvaMorphologySubmit',
            function='fixture_morphology_dilate3x3'+('' if dtype=='u8' else '_'+dtype))
       for dtype,ctype,minimum in [('u8','uint8_t','0'),('s8','int8_t','INT8_MIN'),
                                  ('u16','uint16_t','0'),('s16','int16_t','INT16_MIN')]]
CASES += [dict(name='histogram_u8_'+dtype,dtype=dtype,ctype=ctype,
               family='ImageHistogram',vendor='pvaImageHistogramSubmit',
               function='fixture_image_histogram_u8'+('' if dtype=='u32' else '_s32'))
          for dtype,ctype in [('u32','uint32_t'),('s32','int32_t')]]
CASES += [dict(name='gaussian_u8',dtype='u8',ctype='uint8_t',
               family='GaussianFilter',vendor='pvaGaussianFilterSubmit',
               function='fixture_gaussian_filter3x3',spec='opencv_gaussian3x3_u8',filter_kind=2)]
PENDING=['BoxFilter','BilateralFilter','HistogramEqualization']


def build(case, output):
    if case['family']=='ImageHistogram':
        return build_histogram(case,output)
    if case['family'] in ('GaussianFilter','BoxFilter'):
        return build_fixed_filter(case,output)
    sys.path[:0]=[str(COMPILER/'scripts/correctness'),str(COMPILER/'scripts/correctness/lib_v2')]
    import recipe
    import spec_match
    dest=output/case['name'];dest.mkdir(parents=True,exist_ok=True)
    source=ROOT/'pva/inputs/image_processing.c'
    env=environment(POLYGEIST_ABI_BACKEND='pva',POLYGEIST_NUMERIC_MODEL='strict')
    os.environ.update(env)
    stages=recipe.raise_c(recipe.PRODUCTION,str(source),case['function'],str(dest/'source'),[],120)
    save(dest/'raise.json',{k:dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                            for k,v in stages.items()})
    if not all(stages[s].ok for s in ['cgeist','raise','debuf']):
        raise RuntimeError(case['name']+': raising failed')
    matched,proof=spec_match.match((dest/'source_debuf.mlir').read_text(),backend='pva',
                                   region=True,region_budget=30)
    (dest/'matched.mlir').write_text(matched);save(dest/'proof.json',proof)
    expected='opencv_morphology_dilate_'+case['dtype']
    if proof['emitted']!=[expected] or re.search(r'\b(?:linalg\.generic|affine\.for|scf\.(?:for|while))\b',matched):
        raise RuntimeError(case['name']+': incomplete imported-specification proof')
    harness=ROOT/'scripts/correctness/pva_morphology_matched_dtype_harness.c'
    env.update(POLYGEIST_MATCH_PROOF=str(dest/'proof.json'),POLYGEIST_BUILD_WORK_DIR=str(dest/'work'),
        POLYGEIST_HARNESS_CFLAGS=' '.join(['-DPVA_MORPH_CTYPE='+case['ctype'],
        '-DPVA_MORPH_FUNCTION='+case['function'],'-DPVA_MORPH_MIN='+case['minimum'],
        '-DPVA_MORPH_DTYPE='+case['dtype']]))
    command([COMPILER/'scripts/correctness/polygeist_build.sh','--target=jetson',
        '--function='+case['function'],'--semantic-mlir='+str(dest/'matched.mlir'),
        '--harness='+str(harness),'-o',dest/'program',source],dest/'build.log',env)
    save(dest/'build.json',dict(case=case,source_sha256=sha(source),harness_sha256=sha(harness),
        proof_sha256=sha(dest/'proof.json'),executable_sha256=sha(dest/'program'),emitted=proof['emitted']))
    print(case['name'],'built',flush=True)


def build_histogram(case, output):
    sys.path[:0]=[str(COMPILER/'scripts/correctness'),str(COMPILER/'scripts/correctness/lib_v2')]
    import recipe
    import spec_match
    dest=output/case['name'];dest.mkdir(parents=True,exist_ok=True)
    original=ROOT/'pva/inputs/image_processing.c'
    source=dest/'fixed_shape.c'
    source.write_text('#define '+case['function']+' __attribute__((always_inline)) original_histogram\n'
        '#include '+json.dumps(str(original))+'\n'
        'void artifact_histogram_fixed(const uint8_t input[192][256], '+case['ctype']+' output[256]) {\n'
        '  original_histogram(192,256,&input[0][0],('+case['ctype']+' (*)[256])output);\n}\n')
    function='artifact_histogram_fixed'
    env=environment(POLYGEIST_ABI_BACKEND='pva',POLYGEIST_NUMERIC_MODEL='strict',
                    POLYGEIST_CGEIST_FLAGS='--no-inline=false')
    os.environ.update(env)
    stages=recipe.raise_c(recipe.PRODUCTION,str(source),function,str(dest/'source'),
                          ['--no-inline=false'],120)
    save(dest/'raise.json',{k:dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                           for k,v in stages.items()})
    if not all(stages[s].ok for s in ['cgeist','raise','debuf']):
        raise RuntimeError(case['name']+': raising failed')
    matched,proof=spec_match.match((dest/'source_debuf.mlir').read_text(),backend='pva',
                                  region=True,region_budget=30)
    (dest/'matched.mlir').write_text(matched);save(dest/'proof.json',proof)
    expected='opencv_histogram_u8_'+case['dtype']
    # Signed/unsigned 32-bit outputs share the same signless MLIR type. Their
    # bounded counts have identical bits; accept either validated typed route.
    allowed={'opencv_histogram_u8_u32','opencv_histogram_u8_s32'}
    if len(proof['emitted'])!=1 or proof['emitted'][0] not in allowed or re.search(
            r'\b(?:linalg\.generic|affine\.for|scf\.(?:for|while))\b',matched):
        raise RuntimeError(case['name']+': incomplete imported-specification proof')
    harness=ROOT/'scripts/correctness/pva_histogram_matched_dtype_harness.c'
    env.update(POLYGEIST_MATCH_PROOF=str(dest/'proof.json'),POLYGEIST_BUILD_WORK_DIR=str(dest/'work'),
        POLYGEIST_HARNESS_CFLAGS=' '.join(['-DPVA_HIST_INPUT_CTYPE=uint8_t',
            '-DPVA_HIST_OUTPUT_CTYPE='+case['ctype'],'-DPVA_HIST_FUNCTION='+function,
            '-DPVA_HIST_ROUTE=u8_'+case['dtype'],'-DPVA_HIST_SHIFT=0','-DPVA_HIST_FIXED_SHAPE=1']))
    command([COMPILER/'scripts/correctness/polygeist_build.sh','--target=jetson',
        '--function='+function,'--semantic-mlir='+str(dest/'matched.mlir'),
        '--harness='+str(harness),'-o',dest/'program',source],dest/'build.log',env)
    save(dest/'build.json',dict(case=case,source_sha256=sha(source),original_source_sha256=sha(original),
        harness_sha256=sha(harness),proof_sha256=sha(dest/'proof.json'),
        executable_sha256=sha(dest/'program'),emitted=proof['emitted'],
        shape=dict(height=192,width=256),scope='Fixed benchmark-shape wrapper; original fixture body unchanged.'))
    print(case['name'],'built',flush=True)


def build_fixed_filter(case,output):
    sys.path[:0]=[str(COMPILER/'scripts/correctness'),str(COMPILER/'scripts/correctness/lib_v2')]
    import recipe
    import spec_match
    dest=output/case['name'];dest.mkdir(parents=True,exist_ok=True)
    original=ROOT/'pva/inputs/image_processing.c'
    source=dest/'fixed_shape.c';function='artifact_filter_fixed'
    source.write_text('#define '+case['function']+' __attribute__((always_inline)) original_filter\n'
        '#include '+json.dumps(str(original))+'\n'
        'void '+function+'(const '+case['ctype']+' input[192][256], '+case['ctype']+' output[192][256]) {\n'
        '  original_filter(192,256,&input[0][0],&output[0][0]);\n}\n')
    env=environment(POLYGEIST_ABI_BACKEND='pva',POLYGEIST_NUMERIC_MODEL='strict',
                    POLYGEIST_CGEIST_FLAGS='--no-inline=false')
    os.environ.update(env)
    stages=recipe.raise_c(recipe.PRODUCTION,str(source),function,str(dest/'source'),
                          ['--no-inline=false'],120)
    save(dest/'raise.json',{k:dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                           for k,v in stages.items()})
    if not all(stages[s].ok for s in ['cgeist','raise','debuf']):
        raise RuntimeError(case['name']+': raising failed')
    matched,proof=spec_match.match((dest/'source_debuf.mlir').read_text(),backend='pva',
                                  region=True,region_budget=30)
    (dest/'matched.mlir').write_text(matched);save(dest/'proof.json',proof)
    if proof['emitted']!=[case['spec']] or re.search(
            r'\b(?:linalg\.generic|affine\.for|scf\.(?:for|while))\b',matched):
        raise RuntimeError(case['name']+': incomplete imported-specification proof')
    harness=ROOT/'scripts/correctness/pva_filter_matched_dtype_harness.c'
    env.update(POLYGEIST_MATCH_PROOF=str(dest/'proof.json'),POLYGEIST_BUILD_WORK_DIR=str(dest/'work'),
        POLYGEIST_HARNESS_CFLAGS=' '.join(['-DPVA_FILTER_CTYPE='+case['ctype'],
            '-DPVA_FILTER_FUNCTION='+function,'-DPVA_FILTER_DTYPE='+case['dtype'],
            '-DPVA_FILTER_KIND='+str(case['filter_kind']),'-DPVA_FILTER_FIXED_SHAPE=1']))
    command([COMPILER/'scripts/correctness/polygeist_build.sh','--target=jetson',
        '--function='+function,'--semantic-mlir='+str(dest/'matched.mlir'),
        '--harness='+str(harness),'-o',dest/'program',source],dest/'build.log',env)
    save(dest/'build.json',dict(case=case,source_sha256=sha(source),original_source_sha256=sha(original),
        harness_sha256=sha(harness),proof_sha256=sha(dest/'proof.json'),
        executable_sha256=sha(dest/'program'),emitted=proof['emitted'],
        shape=dict(height=192,width=256),scope='Fixed benchmark-shape wrapper; original fixture body unchanged.'))
    print(case['name'],'built',flush=True)


def run(case,output):
    dest=output/case['name'];build=json.loads((dest/'build.json').read_text())
    if sha(dest/'program')!=build['executable_sha256']:raise RuntimeError('Executable changed')
    target='/tmp/looplibrarian-artifact-pva/'+case['name']
    if remote.upload(dest/'program',target):raise RuntimeError('Upload failed')
    cmd='chmod +x '+shlex.quote(target)+'; '+remote.runtime_env()+' POLYGEIST_V2_TRACE=1 timeout 120 '+shlex.quote(target)
    rc,log=remote.run(cmd,timeout=140);(dest/'run.log').write_text(log)
    vendor=case['vendor'] in log
    passed=rc==0 and 'EXACT_PASS' in log and 'mismatches=0,max_error=0' in log and vendor
    result=dict(case=case['name'],correctness=passed,returncode=rc,
                vendor_call_observed=vendor,executable_sha256=build['executable_sha256'])
    save(dest/'result.json',result);print(case['name'],'PASS' if passed else 'FAIL',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['build','run','summarize','all'],default='all')
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--cases',nargs='+')
    args=p.parse_args();args.output=args.output.resolve()
    cases=[c for c in CASES if not args.cases or c['name'] in args.cases]
    if not cases or (args.cases and set(args.cases)-{c['name'] for c in cases}):p.error('unknown PVA case')
    if args.stage in ['build','all']:
        for case in cases:build(case,args.output)
    if args.stage in ['run','all']:
        for case in cases:run(case,args.output)
    if args.stage in ['run','summarize','all']:
        results=[json.loads((args.output/c['name']/'result.json').read_text()) for c in cases]
        save(args.output/'summary.json',dict(results=results,passed=sum(r['correctness'] for r in results),
            remaining_operator_families=PENDING,timing_claim=False))
        if not all(r['correctness'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
