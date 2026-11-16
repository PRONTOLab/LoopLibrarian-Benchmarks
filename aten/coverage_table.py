"""Report ATen matching coverage from per-kernel compiler evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_LIBRARIES = {'CUB', 'Thrust', 'CUTLASS'}


def unique_rows(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    result = {row['kernel']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate kernel in {path}')
    return result


def complete(variant):
    counts = variant.get('counts', {})
    return bool(variant.get('emitted') and variant.get('abi_ok') and
                counts.get('linalg') == 0 and counts.get('loops') == 0)


def totals(rows, field):
    matched = [row for row in rows if row[field]]
    return dict(all_kernels=len(rows), matched=len(matched),
                native_library_kernels=sum(row['native_library'] for row in rows),
                native_library_matched=sum(row['native_library'] for row in matched),
                other_matched=sum(not row['native_library'] for row in matched))


def reconcile(root, matching=None):
    root = Path(root)
    inventory = unique_rows(root/'aten/all_cases.csv')
    native = unique_rows(root/'aten/metadata/native_cuda_external_library_calls.csv')
    if set(inventory) != set(native):
        raise ValueError('Inventory and native audit differ')
    selected = None
    if matching is not None:
        matching = Path(matching)
        provenance = json.loads((matching/'provenance.json').read_text())
        selected = set(provenance['selected'])
        if selected - set(inventory):
            raise ValueError('Matching results contain kernels outside the inventory')
        if provenance['arguments'].get('shape_specs'):
            raise ValueError('Table 5 requires source fixture dimensions')
    rows = []
    for name, source in inventory.items():
        audit = native[name]
        libraries = audit['external_libraries'].split('|')
        row = dict(kernel=name, native_library=audit['audit_status']=='EXTERNAL_LIBRARY',
                   native_libraries=audit['external_libraries'],
                   templated_native=bool(TEMPLATE_LIBRARIES.intersection(libraries)),
                   current_attempted=False, current_recognized=False,
                   current_abi_ok=False, current_complete=False,
                   current_emitted='', current_status='not_evaluated')
        path = matching/name/'result.json' if matching is not None else None
        if path is not None and path.exists():
            if name not in selected:
                raise ValueError(f'Unexpected result outside saved selection: {name}')
            result = json.loads(path.read_text())
            digest = hashlib.sha256((root/source['source']).read_bytes()).hexdigest()
            if result['source_sha256'] != digest:
                raise ValueError(f'Source differs from matching evidence: {name}')
            variants = result.get('variants', [])
            accepted = [v for v in variants if v.get('emitted') and v.get('abi_ok')]
            best = next((v for v in accepted if complete(v)), accepted[0] if accepted else {})
            row.update(current_attempted=True,
                       current_recognized=any(v.get('emitted') for v in variants),
                       current_abi_ok=bool(accepted), current_complete=any(map(complete, variants)),
                       current_emitted=';'.join(best.get('emitted', [])),
                       current_status=result.get('status', 'incomplete'))
        rows.append(row)
    templated = [row for row in rows if row['native_library'] and row['templated_native']]
    summary = dict(
        current=totals(rows, 'current_complete'),
        current_attempted=sum(row['current_attempted'] for row in rows),
        current_recognized=sum(row['current_recognized'] for row in rows),
        current_abi_ok=sum(row['current_abi_ok'] for row in rows),
        incomplete_matches=sum(r['current_attempted'] and not r['current_complete']
                               for r in rows),
        not_evaluated=sum(not r['current_attempted'] for r in rows),
        templated_native=len(templated),
        matched_templated=[dict(kernel=r['kernel'], emitted=r['current_emitted'])
                           for r in templated if r['current_complete']],
        scope='Complete matching requires imported-reference proof, valid API arguments, and no residual computation loops. GPU correctness and performance are reported separately.')
    summary['current_full_inventory'] = summary['current_attempted'] == len(rows)
    return dict(summary=summary, cases=rows)


def write_report(out, matching=None, root=ROOT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = reconcile(root, matching)
    (out/'table5.json').write_text(json.dumps(report, indent=2)+'\n')
    with (out/'table5.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['cases'][0]))
        writer.writeheader()
        writer.writerows(report['cases'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matching', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report(args.output, args.matching)['summary'], indent=2))


if __name__ == '__main__':
    main()
