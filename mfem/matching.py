"""Compiler-only MFEM matching, with separate contraction and vector counts."""
import csv
import json
from collections import Counter
from pathlib import Path
from common import ROOT, COMPILER, OUT, save

# These are comparison values, never inputs to compiler recognition.
PUBLISHED = {
    'interp_value': 5, 'integrate_value': 3,
    'interp_grad': 12, 'integrate_grad': 13,
    'mass_apply': 8, 'diffusion_apply': 20, 'convection_apply': 15,
    'curlcurl_apply': 34, 'divdiv_apply': 18,
}
CONTRACTION_APIS = {'cublasDgemm', 'cublasDgemv', 'cutensornetNetworkContract',
                   'cutensornetContraction2_f64'}


def family(name):
    return next((key for key in PUBLISHED if name.startswith(key+'_')), name)


def summarize(rows):
    historical = json.loads((ROOT/'mfem/published-matching.json').read_text())
    paper = {r['case']: r for r in historical['results']}
    targets = {}
    for path in (COMPILER/'generic_solver/lib_v3/manifest').glob('*.json'):
        for spec in json.loads(path.read_text()).get('specs', []):
            targets[spec['name']] = spec['target']
    results, groups = [], {}
    for row in rows:
        path = OUT/'matching'/row['kernel']/'full/result.json'
        result = json.loads(path.read_text())
        operations = [dict(spec=name, library=targets[name].get('library'),
                           call=targets[name].get('call')) for name in result['emitted']]
        contractions = [op for op in operations if op['call'] in CONTRACTION_APIS]
        record = dict(case=row['kernel'], family=family(row['kernel']),
                      raised=result['raised'], recognized=result['recognized'],
                      abi_ok=result['abi_ok'], error=result['error'],
                      emitted_regions=len(operations), contraction_regions=len(contractions),
                      operations=operations,
                      evidence=str(path.relative_to(OUT)))
        if row['kernel'] in paper:
            record['paper_regions'] = paper[row['kernel']]['matched_regions']
            record['contraction_count_difference'] = len(contractions)-record['paper_regions']
            record['paper_target'] = paper[row['kernel']]['target']
        results.append(record)
        if record['family'] in PUBLISHED:
            group = groups.setdefault(record['family'], dict(
                current_contraction_regions=0, paper_sites=PUBLISHED[record['family']],
                libraries=Counter(), cases=[]))
            group['cases'].append(record['case'])
            if record['abi_ok']:
                group['current_contraction_regions'] += len(contractions)
                group['libraries'].update(op['library'] for op in contractions)
    pcg = [op for r in results if r['case']=='pcg_application' and r['abi_ok'] for op in r['operations']
           if op['call'] in {'cublasDaxpy', 'cublasDscal', 'cublasDgeam'}]
    report = dict(results=results, by_family=groups,
                  recognized_inputs=sum(r['recognized'] for r in results),
                  abi_lowered_inputs=sum(r['abi_ok'] for r in results),
                  current_contraction_regions=sum(g['current_contraction_regions'] for g in groups.values()),
                  paper_contraction_sites=128, pcg_vector_calls=pcg,
                  paper_pcg_updates=2, source_site_mapping_verified=False,
                  historical_provenance={k:v for k,v in historical.items() if k!='results'},
                  scope='Imported-specification matching and ABI lowering only; no GPU execution or timing. Emitted regions are not automatically the same counting unit as paper source sites.')
    save(OUT/'matching-summary.json', report)
    with (OUT/'matching-table.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['family','paper_sites','current_contraction_regions','libraries'])
        writer.writeheader()
        for name, group in groups.items():
            writer.writerow(dict(family=name, paper_sites=group['paper_sites'],
                current_contraction_regions=group['current_contraction_regions'],
                libraries=json.dumps(group['libraries'], sort_keys=True)))
    return report


def reproduce(rows, contraction_library='cutensornet'):
    from coverage import campaign
    inputs = [dict(name=row['kernel'], source=str(ROOT/'mfem/inputs'/row['source']),
                   function=row['function'],
                   flags=(['-D'+row['macro']] if row.get('macro') else []) +
                         ['-DMFEM_BENCH_NE='+('2' if row['kernel']=='pcg_application' else '1024')] +
                         # The application entry point calls the operator and
                         # PCG helpers. Match their inlined bodies, as in the
                         # execution build, rather than opaque declarations.
                         (['--no-inline=false'] if row['kernel']=='pcg_application' else []),
                   match_options=dict(max_parallel_dims=2, max_operand_rank=2,
                       disabled=(['cublasDgemm', 'cublasDgemv']
                                 if contraction_library == 'cutensornet' else
                                 ['cutensornetNetworkContract'])))
              for row in rows]
    campaign(inputs, OUT/'matching')
    return summarize(rows)
