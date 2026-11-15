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
import local_execution

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
CASES += [dict(name='box_u8',dtype='u8',ctype='uint8_t',family='BoxFilter',
               vendor='pvaBoxFilterSubmit',function='fixture_box_filter3x3',
               spec='opencv_box3x3_u8',filter_kind=1,error_budget=3)]
for dtype,ctype in [('s8','int8_t'),('u16','uint16_t'),('s16','int16_t')]:
    for name,family,vendor,kind in [('box','BoxFilter','pvaBoxFilterSubmit',1),
                                    ('gaussian','GaussianFilter','pvaGaussianFilterSubmit',2)]:
        CASES.append(dict(name=name+'_'+dtype,dtype=dtype,ctype=ctype,family=family,
            vendor=vendor,function='fixture_'+name+'_filter3x3_'+dtype,
            spec='opencv_'+name+'3x3_'+dtype,filter_kind=kind,approximate=True,image_harness=True))
for dtype,ctype in [('u32','uint32_t'),('s32','int32_t')]:
    CASES.append(dict(name='histogram_u16_'+dtype,dtype=dtype,ctype=ctype,
        family='ImageHistogram',vendor='pvaImageHistogramSubmit',input_type='uint16_t',
        input_dtype='u16',shift=8,function='fixture_image_histogram_u16'+('' if dtype=='u32' else '_s32')))
CASES += [dict(name='bilateral_u8',dtype='u8',ctype='uint8_t',family='BilateralFilter',
    vendor='pvaBilateralFilterSubmit',function='fixture_bilateral_filter3x3',
    spec='opencv_bilateral3x3_square_u8',approximate=True,image_harness=True,bilateral=True,dynamic_shape=True),
    dict(name='equalization_u8',dtype='u8',ctype='uint8_t',family='HistogramEqualization',
    vendor='pvaHistogramEqualizationSubmit',function='fixture_histogram_equalization',
    spec='opencv_equalize_integer_u8',approximate=True,image_harness=True,dynamic_shape=True)]
PENDING=[]


def check_build_dependencies():
    env=environment()
    if env.get('PVASOL_ROOT'):
        for key,relative in [('PVA_OPERATOR_INCLUDE','public/src/operator/include'),
                             ('NVCV_INCLUDE','public/3rdparty/cvcuda/src/nvcv/src/include')]:
            if not env.get(key):env[key]=str(Path(env['PVASOL_ROOT'])/relative)
    requirements={
        'PVA_OPERATOR_INCLUDE': 'OpBoxFilter.h',
        'NVCV_INCLUDE': 'nvcv/Tensor.h',
        'CUPVA_SDK_ROOT': 'include/cupva_host.h',
        'PVA_LIB_STAGE': 'libpva_operator.so',
    }
    missing=[key+' ('+relative+')' for key,relative in requirements.items()
             if not env.get(key) or not (Path(env[key])/relative).is_file()]
    if missing:
        raise RuntimeError('Missing installed PVA SDK inputs: '+', '.join(missing)+
                           '. See README.md#pva-table-7 for native and Docker setup.')
    return env


def build_vendor_comparison(output):
    env=check_build_dependencies()
    cuda=Path(env.get('POLYGEIST_CUDA_CROSS_ROOT',env.get('CUDA','/usr/local/cuda')))
    sources=[ROOT/'pva/vendor_compatibility.c',ROOT/'pva/inputs/image_processing.c',
             COMPILER/'runtime/polygeist_pva_image_rt.c']
    executable=output/'vendor-comparison'
    command(['aarch64-linux-gnu-gcc','-O2',*sources,
        '-I'+str(COMPILER/'runtime'),'-I'+str(cuda/'include'),
        '-I'+env['PVA_OPERATOR_INCLUDE'],'-I'+env['NVCV_INCLUDE'],
        '-I'+str(Path(env['CUPVA_SDK_ROOT'])/'include'),
        '-L'+env['PVA_LIB_STAGE'],'-L'+str(cuda/'lib'),
        '-lpva_operator','-lnvcv_types','-lcupva_host','-lcudart',
        '-lm','-lpthread','-ldl','-Wl,--allow-shlib-undefined','-o',executable],
        output/'vendor-build.log',env)
    save(output/'vendor-build.json',dict(executable_sha256=sha(executable),
        source_sha256={str(p):sha(p) for p in sources},
        scope='Direct vendor API comparison; not a compiler matching result.'))


def vendor_comparison_result(rc,log):
    rows=[]
    for line in log.splitlines():
        if line.startswith('COMPARE,'):
            fields=line.split(',')
            rows.append(dict(family=fields[1],pattern=fields[2],
                **{k:int(v) for k,v in (f.split('=') for f in fields[3:])}))
    expected={(family,pattern) for family in ('BoxFilter','BilateralFilter','HistogramEqualization')
              for pattern in ('original_filter','random','impulse','constant','quantization_extremes')}
    complete=(rc==0 and len(rows)==len(expected) and
        {(r['family'],r['pattern']) for r in rows}==expected and
        all(r['count']==192*256 and r['guardbad']==0 and r['inputbad']==0 for r in rows))
    return dict(returncode=rc,execution_complete=complete,comparisons=rows,
                scope='Direct vendor API comparison; differences do not count as compiler matches.')


def run_vendor_comparison(output):
    local_execution.require_aarch64()
    build=json.loads((output/'vendor-build.json').read_text())
    executable=output/'vendor-comparison'
    if sha(executable)!=build['executable_sha256']:raise RuntimeError('Vendor executable changed')
    rc,log=local_execution.run(local_execution.runtime_env()+' timeout 180 '+
                              shlex.quote(str(executable)),timeout=200)
    (output/'vendor-run.log').write_text(log)
    result=vendor_comparison_result(rc,log)
    result['executable_sha256']=build['executable_sha256']
    save(output/'vendor-result.json',result)
    if not result['execution_complete']:raise RuntimeError('Incomplete vendor comparison')


def build(case, output):
    # Compiler modules capture dependency locations when imported. Initialize
    # those settings before either matching or the subprocess ABI validation.
    os.environ.update(environment())
    if case['family']=='ImageHistogram':
        return build_histogram(case,output)
    if case['family'] in ('GaussianFilter','BoxFilter','BilateralFilter','HistogramEqualization'):
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
        'void artifact_histogram_fixed(const '+case.get('input_type','uint8_t')+' input[192][256], '+case['ctype']+' output[256]) {\n'
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
    prefix='opencv_histogram_'+case.get('input_dtype','u8')+'_'
    # Signed/unsigned 32-bit outputs share the same signless MLIR type. Their
    # bounded counts have identical bits; accept either validated typed route.
    allowed={prefix+'u32',prefix+'s32'}
    if len(proof['emitted'])!=1 or proof['emitted'][0] not in allowed or re.search(
            r'\b(?:linalg\.generic|affine\.for|scf\.(?:for|while))\b',matched):
        raise RuntimeError(case['name']+': incomplete imported-specification proof')
    harness=ROOT/'scripts/correctness/pva_histogram_matched_dtype_harness.c'
    env.update(POLYGEIST_MATCH_PROOF=str(dest/'proof.json'),POLYGEIST_BUILD_WORK_DIR=str(dest/'work'),
        POLYGEIST_HARNESS_CFLAGS=' '.join(['-DPVA_HIST_INPUT_CTYPE='+case.get('input_type','uint8_t'),
            '-DPVA_HIST_OUTPUT_CTYPE='+case['ctype'],'-DPVA_HIST_FUNCTION='+function,
            '-DPVA_HIST_ROUTE='+case.get('input_dtype','u8')+'_'+case['dtype'],'-DPVA_HIST_SHIFT='+str(case.get('shift',0)),'-DPVA_HIST_FIXED_SHAPE=1']))
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
        '  original_filter(192,256,&input[0][0],&output[0][0]'+(',25.0f,1.5f' if case.get('bilateral') else '')+');\n}\n')
    if case.get('dynamic_shape'):
        parameters=',float sigma_r,float sigma_s' if case.get('bilateral') else ''
        arguments=',sigma_r,sigma_s' if case.get('bilateral') else ''
        source.write_text('#define '+case['function']+' __attribute__((always_inline)) original_filter\n'
            '#include '+json.dumps(str(original))+'\n'
            'void '+function+'(int h,int w,const '+case['ctype']+' input[h*w],'
            +case['ctype']+' output[h*w]'+parameters+') {\n'
            '  original_filter(h,w,input,output'+arguments+');\n}\n')
    env=environment(POLYGEIST_ABI_BACKEND='pva',
                    POLYGEIST_NUMERIC_MODEL='relaxed' if case.get('approximate') else 'strict',
                    POLYGEIST_BACKEND_APPROXIMATIONS=json.dumps([case['spec']] if case.get('approximate') else []),
                    POLYGEIST_CGEIST_FLAGS='--no-inline=false',
                    POLYGEIST_BACKEND_ERROR_BUDGETS=json.dumps(
                        {case['spec']:case['error_budget']} if case.get('error_budget') else {}))
    os.environ.update(env)
    stages=recipe.raise_c(recipe.PRODUCTION,str(source),function,str(dest/'source'),
                          ['--no-inline=false'],120)
    save(dest/'raise.json',{k:dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                           for k,v in stages.items()})
    if not all(stages[s].ok for s in ['cgeist','raise','debuf']):
        raise RuntimeError(case['name']+': raising failed')
    # Both stages use the common raising recipe. Keep the memory form when
    # tensor conversion prevents a complete proof of a mixed structured region.
    for stage in ('debuf','linalg'):
        matched,proof=spec_match.match((dest/('source_'+stage+'.mlir')).read_text(),
                                      backend='pva',region=True,region_budget=30)
        if proof['emitted'] and not re.search(
                r'\b(?:linalg\.generic|affine\.for|scf\.(?:for|while))\b',matched):
            break
    (dest/'matched.mlir').write_text(matched);save(dest/'proof.json',proof)
    if proof['emitted']!=[case['spec']] or re.search(
            r'\b(?:linalg\.generic|affine\.for|scf\.(?:for|while))\b',matched):
        raise RuntimeError(case['name']+': incomplete imported-specification proof')
    if case.get('image_harness'):
        harness=ROOT/'scripts/correctness/pva_image_matched_harness.c'
        flags=['-DPVA_IMAGE_CTYPE='+case['ctype'],'-DPVA_IMAGE_REFERENCE='+case['function'],
               '-DPVA_IMAGE_FUNCTION='+function,'-DPVA_IMAGE_BILATERAL='+str(int(case.get('bilateral',False))),
               '-DPVA_IMAGE_DYNAMIC_SHAPE='+str(int(case.get('dynamic_shape',False))),
               '-DPVA_IMAGE_PRESERVE_BORDER='+str(int(case['family']!='HistogramEqualization'))]
    else:
        harness=ROOT/'scripts/correctness/pva_filter_matched_dtype_harness.c'
        flags=['-DPVA_FILTER_CTYPE='+case['ctype'],'-DPVA_FILTER_FUNCTION='+function,
               '-DPVA_FILTER_DTYPE='+case['dtype'],'-DPVA_FILTER_KIND='+str(case['filter_kind']),
               '-DPVA_FILTER_FIXED_SHAPE=1','-DPVA_FILTER_TOLERANCE='+str(case.get('error_budget',0))]
    env.update(POLYGEIST_MATCH_PROOF=str(dest/'proof.json'),POLYGEIST_BUILD_WORK_DIR=str(dest/'work'),
               POLYGEIST_HARNESS_CFLAGS=' '.join(flags))
    command([COMPILER/'scripts/correctness/polygeist_build.sh','--target=jetson',
        '--function='+function,'--semantic-mlir='+str(dest/'matched.mlir'),
        '--harness='+str(harness),'-o',dest/'program',source],dest/'build.log',env)
    save(dest/'build.json',dict(case=case,source_sha256=sha(source),original_source_sha256=sha(original),
        harness_sha256=sha(harness),proof_sha256=sha(dest/'proof.json'),
        executable_sha256=sha(dest/'program'),emitted=proof['emitted'],
        backend_contracts=proof.get('backend_contracts',{}),
        shape=dict(height=192,width=256),matching_stage=stage,
        scope=('Original fixture with runtime dimensions; body unchanged.' if case.get('dynamic_shape')
               else 'Fixed benchmark-shape wrapper; original fixture body unchanged.')))
    print(case['name'],'built',flush=True)


def check_output(case,pattern,rc,log):
    vendor=case['vendor'] in log
    match=re.search(r'max_error=(\d+)',log)
    error=int(match[1]) if match else None
    passed=rc==0 and vendor and error is not None and error<=case.get('error_budget',0)
    if case.get('image_harness'):
        passed=(rc==0 and vendor and error is not None and 'IMAGE_RAISED' in log and
                'input_unchanged=1,border_unchanged=1,guards_ok=1' in log and
                re.search(r',input='+re.escape(pattern)+r'(?:\s|$)',log) is not None)
    elif 'filter_kind' in case:
        passed=passed and 'input_unchanged=1,border_unchanged=1,guards_ok=1' in log
        passed=passed and re.search(r',input='+re.escape(pattern)+r'(?:\s|$)',log) is not None
        passed=passed and ('EXACT_PASS' in log or 'WITHIN_TOLERANCE' in log)
    else:
        passed=passed and 'EXACT_PASS' in log and 'mismatches=0,max_error=0' in log
    return dict(input=pattern,correctness=passed,returncode=rc,
                max_abs_error=error,vendor_call_observed=vendor)


def run(case,output):
    local_execution.require_aarch64()
    dest=output/case['name'];build=json.loads((dest/'build.json').read_text())
    if sha(dest/'program')!=build['executable_sha256']:raise RuntimeError('Executable changed')
    if build['case']!=case:raise RuntimeError('Build configuration changed; rebuild the case')
    target='/tmp/looplibrarian-artifact-pva/'+case['name']
    if local_execution.stage(dest/'program',target):raise RuntimeError('Local staging failed')
    checks=[]
    patterns=['original','random','impulse','constant','quantization'] if 'filter_kind' in case or case.get('image_harness') else ['original']
    for pattern in patterns:
        cmd=('chmod +x '+shlex.quote(target)+'; '+local_execution.runtime_env()+
             ' POLYGEIST_V2_TRACE=1 PVA_FILTER_INPUT_PATTERN='+pattern+
             ' timeout 120 '+shlex.quote(target))
        rc,log=local_execution.run(cmd,timeout=140);(dest/('run-'+pattern+'.log')).write_text(log)
        if pattern=='original':(dest/'run.log').write_text(log)
        checks.append(check_output(case,pattern,rc,log))
    passed=all(c['correctness'] for c in checks)
    result=dict(case=case['name'],correctness=passed,
                returncode=next((c['returncode'] for c in checks if c['returncode']),0),
                vendor_call_observed=all(c['vendor_call_observed'] for c in checks),
                numerical_criterion=('recorded-approximation' if case.get('approximate') else
                                     'bounded-quantization' if case.get('error_budget') else 'exact'),
                numerical_exact=all(c['max_abs_error']==0 for c in checks),
                error_budget=case.get('error_budget',0),checks=checks,
                backend_contracts=build.get('backend_contracts',{}),
                executable_sha256=build['executable_sha256'])
    save(dest/'result.json',result);print(case['name'],'PASS' if passed else 'FAIL',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['build','run','summarize','all'],default='all')
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--cases',nargs='+')
    p.add_argument('--vendor-compatibility',action='store_true',
                   help='Also compare Box, Bilateral and HistogramEqualization APIs directly with the fixtures')
    args=p.parse_args();args.output=args.output.resolve()
    cases=[c for c in CASES if not args.cases or c['name'] in args.cases]
    if not cases or (args.cases and set(args.cases)-{c['name'] for c in cases}):p.error('unknown PVA case')
    if args.stage in ['build','all']:
        check_build_dependencies()
        for case in cases:build(case,args.output)
        if args.vendor_compatibility:build_vendor_comparison(args.output)
    if args.stage in ['run','all']:
        for case in cases:run(case,args.output)
        if (args.output/'vendor-build.json').exists():run_vendor_comparison(args.output)
    if args.stage in ['run','summarize','all']:
        results=[json.loads((args.output/c['name']/'result.json').read_text()) for c in cases]
        save(args.output/'summary.json',dict(results=results,passed=sum(r['correctness'] for r in results),
            remaining_operator_families=PENDING,timing_claim=False))
        if not all(r['correctness'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
