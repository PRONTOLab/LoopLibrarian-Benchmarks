#!/usr/bin/env python3
"""Compare runtime reuse in an isolated compiler image; do not edit the checkout.

Example (the SDK layout is the same as the Orin Docker build):
  python3 compilation_cost/runtime_reuse_probe.py --sdk-root /path/to/sdk \
    --config config.docker.json --output results/runs/compilation-reuse

The control and experimental builds compile the application and verify its
matches afresh. Only library metadata and shared runtime objects are reused.
This is a diagnostic experiment, not the complete Table 8 measurement.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

IMAGE = 'ghcr.io/prontolab/looplibrarian-benchmarks:artifact-5903d7b1adb3'
DEFAULT_CASES = ['polybench/gemm', 'aten/aten_avg_pool2d']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(command, **kwargs):
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Pinned build script changed: ' + old[:100])
    return text.replace(old, new, 1)


def runtime_patch(text):
    # ccache is used ONLY at shared-runtime compile sites. The application,
    # harness, wrapper and link commands keep their original compilers.
    text = replace_once(text, 'SPEC_RUNTIME_OBJECTS=()', '''
probe_runtime_cc() {
  env -u CCACHE_DISABLE /usr/bin/ccache "$@"
}
SPEC_RUNTIME_OBJECTS=()''')
    text = replace_once(text,
        '--output-dir "$WORK/spec_runtime" \\\n',
        '--output-dir "$WORK/spec_runtime" --required-ir "$WORK/abi_canon.mlir" \\\n')
    # Change only compiler invocations whose continuation contains a runtime
    # source. A pinned source count makes unnoticed drift an explicit failure.
    lines = text.splitlines(keepends=True)
    changed = 0
    for i, line in enumerate(lines):
        if not re.match(r'\s*(?:\$CC|"\$\{POLYGEIST_NVCC:)', line):
            continue
        command = line
        j = i
        while lines[j].rstrip().endswith('\\'):
            j += 1
            command += lines[j]
        if re.search(r'-c (?:"?\$RT_SRC|"?\$RT/|"\$WORK/spec_runtime/)', command):
            lines[i] = re.sub(r'^(\s*)', r'\1probe_runtime_cc ', line, count=1)
            changed += 1
    if changed < 7:
        raise RuntimeError(f'Only {changed} runtime compiler sites found')
    text = ''.join(lines)
    old = '''if { [ "$TARGET" = jetson ] || [ "$TARGET" = cuda ]; } && [ "$ABI_BACKEND" = cuda ]; then
  probe_runtime_cc "${POLYGEIST_NVCC:-/usr/local/cuda-12.6/bin/nvcc}"'''
    new = '''if { [ "$TARGET" = jetson ] || [ "$TARGET" = cuda ]; } && [ "$ABI_BACKEND" = cuda ] &&
   { grep -Eq 'polygeist_thrust_|polygeist_aten_special_' "$WORK/spec_runtime/polygeist_v2_rt_cuda.c" 2>/dev/null ||
     grep -Eq 'call[[:space:]]+@polygeist_v2_memref_copy\\(' "$WORK/abi_canon.mlir"; }; then
  probe_runtime_cc "${POLYGEIST_NVCC:-/usr/local/cuda-12.6/bin/nvcc}"'''
    return replace_once(text, old, new)


def generator_patch(text):
    text = replace_once(text, '    a = ap.parse_args()',
                        '    ap.add_argument("--required-ir", type=Path)\n    a = ap.parse_args()')
    return replace_once(text, '    new = {HDR:', '''    if a.required_ir:
        required = set(re.findall(r'\\bcall\\s+@polygeist_v2_([\\w.$]+)\\(', a.required_ir.read_text()))
        # This layout-copy helper is supplied by the unchanged runtime include,
        # not by the library ABI generator. All other unknown calls are errors.
        required.discard('memref_copy')
        missing = required - set(calls)
        if missing:
            raise SystemExit('Missing runtime bindings: ' + repr(sorted(missing)))
        calls = {name: value for name, value in calls.items() if name in required}
    new = {HDR:''')


# Cache only the dataclass metadata returned by build_table. Cached files are
# private, locally produced, never downloaded. The key is supplied by this
# experiment's content manifest. Every saved input is checked before a build;
# the source tree and SDK cannot be changed by another experiment process.
ABI_CACHE = '''
_probe_uncached_build_table = build_table
def build_table(specs=None):
    import hashlib, pickle
    directory = os.environ.get('LL_PROBE_ABI_CACHE')
    if not directory:
        return _probe_uncached_build_table(specs)
    global MANIFEST
    MANIFEST = _manifest_entries()
    key = hashlib.sha256((os.environ['LL_PROBE_INPUT_KEY'] +
                          repr(specs)).encode()).hexdigest()
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / (key + '.pickle')
    if path.exists():
        with path.open('rb') as stream:
            result = pickle.load(stream)
        event = 'hit'
    else:
        result = _probe_uncached_build_table(specs)
        temporary = folder / (key + '.' + str(os.getpid()) + '.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(result, stream)
        temporary.replace(path)
        event = 'miss'
    with (folder / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=event, pid=os.getpid(), key=key))+'\\n')
    return result

'''


def fingerprint(root):
    files = []
    for folder, suffixes in [('scripts/correctness', {'.py', '.sh'}),
                            ('runtime', {'.c', '.h', '.cu', '.cpp', '.inc'}),
                            ('generic_solver/lib_v3', {'.json', '.c', '.mlir', '.h'})]:
        files += [p for p in (root/folder).rglob('*') if p.suffix in suffixes
                  and p.is_file() and 'validate' not in p.parts]
    files += [root/'generic_solver/kernel_library_v3.mlir',
              root/'build/bin/cgeist', root/'build/bin/polygeist-opt']
    # Include all selected library source/header inputs named by the imported
    # reference provenance, including public dependencies outside Polygeist.
    provenance = json.loads((root/'generic_solver/lib_v3/import_manifest.json').read_text())
    dependencies = Path(os.environ.get('POLYGEIST_DEPENDENCY_ROOT', root/'third_party'))
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and re.fullmatch('[a-f0-9]{64}', item):
                    candidate = (dependencies/Path(key).relative_to('third_party')
                                 if key.startswith('third_party/') else root/key)
                    if candidate.is_file(): files.append(candidate)
                else: visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    visit(provenance)
    files += [p for p in (dependencies/'cuda_headers').rglob('*') if p.is_file()]
    netlib = Path(os.environ.get('NETLIB_REF', '/home/arjaiswal/netlib_ref/lapack'))
    if netlib.exists(): files += list(netlib.rglob('*.f'))
    return {str(p): sha(p) for p in sorted(set(files))}


def app_code(work):
    result = run(['aarch64-linux-gnu-objdump', '-dr', work/'kernel.o'], capture_output=True, text=True)
    text = '\n'.join(line for line in result.stdout.splitlines() if 'file format' not in line)
    return hashlib.sha256(text.encode()).hexdigest()


def ccache_stats():
    return run(['/usr/bin/ccache', '--show-stats', '--verbose'],
               capture_output=True, text=True).stdout


def wrapper_body(text, symbol):
    match = re.search(r'\b' + re.escape(symbol) + r'\([^;{}]*\)\s*\{', text)
    if not match: raise RuntimeError('Missing generated wrapper ' + symbol)
    start = match.start(); end = match.end(); depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def summarize(report):
    controls = {row['case']: row for row in report['results'] if row['phase'] == 'control'}
    lines = ['Compilation runtime reuse probe (seconds)', '']
    for name, control in controls.items():
        lines.append(name)
        for row in report['results']:
            if row['case'] != name:
                continue
            ratio = control['seconds']/row['seconds']
            lines.append(f"  {row['phase']:16s} total {row['seconds']:8.2f}  "
                         f"runtime {row['runtime_seconds']:8.2f}  {ratio:.2f}x vs control")
        lines.append('')
    lines += [
        'One build per input and phase; this is not the complete Table 8 study.',
        'Each phase starts a new parsed-specification cache; its second input reuses it.',
        'Runtime objects persist between experimental phases. ABI metadata persists',
        'between metadata_cold and metadata_warm. Cold phases populate these caches.',
        'Input hashing is recorded separately in phase/setup.json.',
        report['validation'],
    ]
    return '\n'.join(lines) + '\n'


def inside(args):
    root = Path('/opt/looplibrarian')
    compiler = root/'third_party/Polygeist'
    os.environ.update(LL_ARTIFACT_CONFIG='/out/config.json', LL_GPU_TARGET='jetson',
                      CCACHE_DIR='/out/object-cache', CCACHE_COMPILERCHECK='content',
                      CCACHE_BASEDIR='/out', CCACHE_NOHASHDIR='true', CCACHE_MAXSIZE='2G')
    sys.path[:0] = [str(root/'scripts/artifact'), str(root/'scripts/correctness'),
                    str(compiler/'scripts/correctness'), str(compiler/'scripts/correctness/lib_v2')]
    spec = importlib.util.spec_from_file_location('cost_driver', root/'compilation_cost/reproduce.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    os.environ.update(driver.environment())
    import spec_provenance
    errors, error = spec_provenance.validate(compiler/'generic_solver/lib_v3/import_manifest.json')
    stale = {k:v for k,v in errors.items() if not v.startswith('reference import has no ')}
    if error or stale: raise RuntimeError('Invalid imports: ' + str(error or stale))
    available = json.loads((root/'compilation_cost/inputs.json').read_text())['cases']
    cases = [c for name in args.cases for c in available if c['suite']+'/'+c['kernel']==name]
    if len(cases) != len(args.cases): raise RuntimeError('Unknown or duplicate cases')
    build_path = compiler/'scripts/correctness/polygeist_build.sh'
    generator_path = compiler/'scripts/correctness/lib_v2/gen_v2_runtime.py'
    abi_path = compiler/'scripts/correctness/lib_v2/abi_v2.py'
    originals = {p:p.read_text() for p in (build_path, generator_path, abi_path)}
    out = Path('/out'); reports = []; controls = {}; snapshots = {}
    phases = ['control', 'runtime_cold', 'runtime_warm', 'metadata_cold', 'metadata_warm']
    try:
        for phase in phases:
            if phase == 'runtime_cold':
                build_path.write_text(runtime_patch(originals[build_path]))
                generator_path.write_text(generator_patch(originals[generator_path]))
            if phase == 'metadata_cold':
                abi_path.write_text(replace_once(originals[abi_path], '\ndef main() -> int:',
                                                 '\n'+ABI_CACHE+'\ndef main() -> int:'))
                os.environ['LL_PROBE_ABI_CACHE'] = '/out/metadata-cache'
            started = time.monotonic()
            inputs = fingerprint(compiler)
            key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
            os.environ['LL_PROBE_INPUT_KEY'] = key
            previous_inputs = out/phase/'inputs.json'
            if previous_inputs.exists() and json.loads(previous_inputs.read_text()) != inputs:
                raise RuntimeError('Reused control has different compiler inputs')
            save(out/phase/'inputs.json', inputs)
            # Immutability is checked without rereading gigabytes before every
            # build. The content digest above is retained for independent audit.
            state = {Path(p):(Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in inputs}
            save(out/phase/'setup.json', dict(seconds=time.monotonic()-started, input_key=key))
            for case in cases:
                before = ccache_stats()
                options = SimpleNamespace(output=out/phase, timeout=args.timeout, compiler_state=state)
                driver.measure(case, 'full', 1, options)
                directory = out/phase/case['suite']/case['kernel']/'full-1'
                result = json.loads((directory/'result.json').read_text())
                if result['status'] != 'ok':
                    raise RuntimeError(f'{phase}/{case["kernel"]}: {result["status"]}')
                work = directory/'work'
                proof = json.loads((work/'match.json').read_text())
                canonical = dict(code_sha256=app_code(work), emitted=result['emitted'],
                                 abi_sha256=proof['abi_sha256'])
                name = case['suite']+'/'+case['kernel']
                wrappers = work/'spec_runtime/polygeist_v2_rt_cuda.c'
                symbols = sorted(set(re.findall(r'\bcall\s+@polygeist_v2_([\w.$]+)\(',
                                                (work/'abi_canon.mlir').read_text())))
                bodies = {s:wrapper_body(wrappers.read_text(), 'polygeist_v2_'+s)
                          for s in symbols if s != 'memref_copy'}
                if 'memref_copy' in symbols:
                    bodies['memref_copy'] = sha(compiler/'runtime/polygeist_v2_memref_copy.inc')
                if phase == 'control':
                    controls[name] = canonical; snapshots[name] = bodies
                elif canonical != controls[name] or bodies != snapshots[name]:
                    save(directory/'mismatch.json', dict(control=controls[name], current=canonical,
                         wrappers_equal=bodies == snapshots[name]))
                    raise RuntimeError('Generated computation/binding changed: '+phase+'/'+name)
                header = run(['readelf','-h',directory/'program'],capture_output=True,text=True).stdout
                if 'AArch64' not in header: raise RuntimeError('Wrong executable architecture')
                stages = {s['label'].split(']')[0]+']': s['duration_seconds'] for s in result['stages']}
                row = dict(phase=phase,case=name,seconds=result['wall_seconds'],
                    matching_seconds=stages.get('[3/9]'),abi_seconds=stages.get('[4-5/9]'),
                    runtime_seconds=stages.get('[8/9]'),emitted_calls=len(result['emitted']),
                    application_code_unchanged=True, selected_wrappers_unchanged=True,
                    executable_architecture='AArch64',
                    thrust_object=(work/'thrust_runtime.o').exists(),
                    source_hash=sha(root/case['source']))
                reports.append(row)
                (directory/'ccache-before.txt').write_text(before)
                (directory/'ccache-after.txt').write_text(ccache_stats())
                save(out/'progress.json',dict(status='running',results=reports))
                print(json.dumps(row),flush=True)
        report = dict(status='complete',results=reports,
            scope='Two-input default diagnostic; one fresh application build per phase. Not Table 8.',
            validation='Generated application machine code, emitted calls, ABI tables and selected wrapper bodies are unchanged. Programs link as AArch64; GPU output execution is not part of this compilation test.',
            cold_cost='runtime_cold and metadata_cold include cache misses. Setup hashing time is recorded separately in each phase/setup.json.',
            policy='Application compilation and match verification are never cached. Shared runtime objects use ccache; ABI tables are reused only within the pinned image/input snapshot.')
        events = [json.loads(line) for line in (out/'metadata-cache/events.jsonl').read_text().splitlines()]
        save(out/'metadata-cache-summary.json',
             {kind: sum(event['event'] == kind for event in events) for kind in ('hit', 'miss')})
        save(out/'report.json',report)
        with (out/'report.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(reports[0]));writer.writeheader();writer.writerows(reports)
        save(out/'progress.json',dict(status='complete',results=reports))
    finally:
        for path, text in originals.items(): path.write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image',default=IMAGE)
    p.add_argument('--sdk-root',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--cases',nargs='+',default=DEFAULT_CASES)
    p.add_argument('--timeout',type=int,default=360)
    p.add_argument('--reuse-control',type=Path,
                   help='Reuse successful control builds from the same pinned image and config')
    p.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.inside:
        try: inside(args)
        except Exception as error:
            path=Path('/out/progress.json')
            progress=json.loads(path.read_text()) if path.exists() else {}
            save(path,dict(progress,status='failed',error=str(error)))
            raise
        return
    if not all((args.sdk_root,args.config,args.output)):p.error('--sdk-root, --config and --output are required')
    if args.output.exists() and any(args.output.iterdir()):p.error('Use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    out=args.output.resolve();sdk=args.sdk_root.resolve()
    shutil.copyfile(args.config,out/'config.json');shutil.copyfile(__file__,out/'probe.py')
    image=run(['docker','image','inspect',args.image,'--format','{{.Id}}'],capture_output=True,text=True).stdout.strip()
    if args.reuse_control:
        old = json.loads((args.reuse_control/'request.json').read_text())
        if (old['image_id'] != image or old['config_sha256'] != sha(out/'config.json')
                or old['cases'] != args.cases):
            p.error('Control image, configuration and cases must agree')
        shutil.copytree(args.reuse_control/'control',out/'control')
        save(out/'reused-control.json',dict(directory=str(args.reuse_control.resolve()),
                                          request=old))
    save(out/'request.json',dict(image=args.image,image_id=image,cases=args.cases,
         script_sha256=sha(Path(__file__)),config_sha256=sha(out/'config.json'),
         host_uptime=run(['uptime'],capture_output=True,text=True).stdout.strip()))
    command=['docker','run','--rm','--mount',f'type=bind,src={sdk},dst=/sdk,readonly',
             '--mount',f'type=bind,src={out},dst=/out',args.image,
             'python3','-u','/out/probe.py','--inside','--timeout',str(args.timeout),'--cases',*args.cases]
    with (out/'driver.log').open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(f'Probe failed; see {out}/driver.log')
    report=json.loads((out/'report.json').read_text())
    summary = summarize(report)
    (out/'summary.txt').write_text(summary)
    print(summary,end='')
    print('Report:',out/'report.csv')


if __name__=='__main__':main()
