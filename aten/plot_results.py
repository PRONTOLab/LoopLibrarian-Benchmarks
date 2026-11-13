#!/usr/bin/env python3
"""Plot validated ATen timings in the paper's log-log scatter format."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess


def band_counts(points):
    ratios = [y / x for x, y in points]
    return dict(timed=len(ratios), more_than_2x_faster=sum(r < .5 for r in ratios),
                within_2x=sum(.5 <= r <= 2 for r in ratios),
                more_than_2x_slower=sum(r > 2 for r in ratios))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True, help='gpu/report.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paper-figure', type=Path,
                        help='Optional archived paper PDF to place beside the current graph unchanged')
    args = parser.parse_args()
    report = args.report.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(report.read_text())
    groups = {'paper': [], 'alternative': []}
    rows = []
    for case in data['cases']:
        runtime = case.get('runtime', {})
        timing = runtime.get('timing', {})
        original = runtime.get('correctness', {}).get('paper', {})
        row = dict(kernel=case['kernel'], status='unmatched', input_kind='',
                   native_us='', raised_us='', raised_over_native='',
                   original_raised_passed=original.get('raised_passed'),
                   original_native_passed=original.get('native_passed'),
                   original_raised_native_agreement=original.get('raised_matches_native'),
                   paper_native_us=case['published']['native_us'],
                   paper_raised_us=case['published']['raised_us'],
                   paper_timing_protocol=case['published']['timing_protocol'])
        if case['complete_match']:
            row['status'] = ('build_failed' if case.get('build', {}).get('rc') != 0
                             else 'comparison_failed_or_unavailable')
        if timing.get('passed'):
            recipe = timing['input_kind']
            assert case['complete_match'] and case['build']['rc'] == 0, case['kernel']
            assert runtime['correctness'][recipe]['passed'], case['kernel']
            assert case['build']['binary_sha256'] == runtime['binary_sha256'], case['kernel']
            minimum = timing['minimum_ms']
            for path in ('raised', 'native_aten_cuda'):
                samples = [s for s in timing['samples'] if s['path'] == path]
                assert len(samples) == 5 and {s['sample'] for s in samples} == set(range(5))
                assert all(math.isfinite(s['ms']) and s['ms'] > 0 for s in samples)
                assert math.isclose(min(s['ms'] for s in samples), minimum[path], rel_tol=1e-10)
            native, raised = minimum['native_aten_cuda'] * 1000, minimum['raised'] * 1000
            group = 'paper' if recipe == 'paper' and runtime['fully_validated'] else 'alternative'
            if group == 'alternative':
                assert recipe == 'native-paper', 'Unclassified timing-input exception'
            groups[group].append((native, raised))
            row.update(status='validated_original_inputs' if group == 'paper' else 'validated_alternative_inputs',
                       input_kind=recipe, native_us=native, raised_us=raised,
                       raised_over_native=raised / native)
        rows.append(row)
    points = groups['paper'] + groups['alternative']
    if not points:
        raise SystemExit('No validated timing pairs to plot')
    assert len(points) == data['summary']['timing_pairs_validated']
    with (out / 'aten-current-data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Include all points; the old 50-us lower limit clipped some new results.
    xlow = 10 ** math.floor(math.log10(min(x for x, y in points)))
    ylow = 10 ** math.floor(math.log10(min(y for x, y in points)))
    low = min(xlow, ylow)
    xhigh = 10 ** math.ceil(math.log10(max(x for x, y in points)))
    yhigh = 10 ** math.ceil(math.log10(max(y for x, y in points)))
    high = max(xhigh, yhigh)
    scatter = []
    for group, label, color, mark in [
            ('paper', 'Original-input checks passed', 'raisedgreen', '*'),
            ('alternative', 'Finite-input checks passed', 'alternateorange', 'diamond')]:
        if not groups[group]:
            continue
        coords = ' '.join(f'({x:.9g},{y:.9g})' for x, y in groups[group])
        fill = color if group == 'paper' else 'white'
        size = '2.3' if group == 'paper' else '3.5'
        scatter.append(rf'\addplot+[only marks,mark={mark},mark size={size}pt,'
                       rf'mark options={{solid,draw={color},fill={fill}}}] '
                       rf'coordinates {{{coords}}};' + '\n' +
                       rf'\addlegendentry{{{label} ({len(groups[group])})}}')
    tex = rf'''\documentclass[tikz,border=5pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\definecolor{{raisedgreen}}{{HTML}}{{1A7F37}}
\definecolor{{boundred}}{{HTML}}{{BD3438}}
\definecolor{{alternateorange}}{{HTML}}{{B35C00}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{loglogaxis}}[
 width=6.65in,height=5.15in,
 title={{ATen: LoopLibrarian vs native CUDA}},title style={{font=\bfseries\large}},
 xlabel={{Native ATen CUDA runtime ($\mu$s)}},
 ylabel={{LoopLibrarian resident GPU runtime ($\mu$s)}},
 xmin={low},xmax={xhigh},ymin={low},ymax={yhigh},
 grid=both,minor grid style={{draw=gray!10}},major grid style={{draw=gray!25}},
 ticklabel style={{font=\small}},label style={{font=\small}},axis lines*=left,
 legend style={{font=\footnotesize,at={{(0.5,-0.19)}},anchor=north,draw=none,
               legend columns=2,column sep=12pt,row sep=3pt}},
]
\addplot+[no marks,draw=black,line width=0.9pt,forget plot]
 coordinates {{({low},{low}) ({high},{high})}};
\addplot+[no marks,draw=boundred,densely dotted,line width=1pt,forget plot]
 coordinates {{({low},{low * 2}) ({high},{high * 2})}};
\addplot+[no marks,draw=boundred,densely dotted,line width=1pt,forget plot]
 coordinates {{({low},{low / 2}) ({high},{high / 2})}};
{chr(10).join(scatter)}
\addlegendimage{{no marks,draw=black,line width=0.9pt}}
\addlegendentry{{Parity ($y=x$)}}
\addlegendimage{{no marks,draw=boundred,densely dotted,line width=1pt}}
\addlegendentry{{$2\times$ and $0.5\times$ native time}}
\draw[->,black,line width=0.9pt] (axis cs:2300,200) -- (axis cs:5000,60)
 node[pos=1,below left,font=\footnotesize,fill=white,inner sep=2pt]
 {{LoopLibrarian faster}};
\end{{loglogaxis}}
\end{{tikzpicture}}
\end{{document}}
'''
    (out / 'aten-current.tex').write_text(tex)
    with (out / 'render.log').open('w') as log:
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                        'aten-current.tex'], cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['pdftoppm', '-png', '-singlefile', '-scale-to', '1500',
                        'aten-current.pdf', 'aten-current'], cwd=out,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = dict(source_report=str(report), source_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   cohort=len(rows), original_inputs=len(groups['paper']),
                   alternative_inputs=len(groups['alternative']), **band_counts(points),
                   unplotted=[dict(kernel=r['kernel'], status=r['status']) for r in rows
                              if not r['status'].startswith('validated_')],
                   timing='Minimum of five synchronized resident samples after five warmups; alternating order.',
                   historical_statistic_note='Paper mixed 58 minimum-of-five and 57 retained best-of-20 raised timings.',
                   axes='Both axes logarithmic; lower raised time relative to parity is better.')
    (out / 'plot-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.paper_figure:
        original = args.paper_figure.resolve().read_bytes()
        (out / 'aten-paper-original.pdf').write_bytes(original)
        # Embed the actual paper figure, preserving its original points, axes,
        # legend and fonts. Panel labels are outside both embedded PDFs.
        comparison_tex = rf'''\documentclass[border=8pt]{{standalone}}
\usepackage{{graphicx}}
\begin{{document}}
\begin{{minipage}}[t]{{6.7in}}
\vspace{{0pt}}\centering
{{\large\bfseries Published paper: {len(rows)} kernels}}\par\vspace{{8pt}}
\includegraphics[width=\linewidth]{{aten-paper-original.pdf}}
\end{{minipage}}\hspace{{0.3in}}
\begin{{minipage}}[t]{{6.7in}}
\vspace{{0pt}}\centering
{{\large\bfseries Current results: {len(points)} validated timing pairs}}\par\vspace{{8pt}}
\includegraphics[width=\linewidth]{{aten-current.pdf}}
\end{{minipage}}
\end{{document}}
'''
        (out / 'aten-paper-vs-current.tex').write_text(comparison_tex)
        with (out / 'comparison-render.log').open('w') as log:
            subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                            'aten-paper-vs-current.tex'], cwd=out, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            subprocess.run(['pdftoppm', '-png', '-singlefile', '-scale-to', '2400',
                            'aten-paper-vs-current.pdf', 'aten-paper-vs-current'], cwd=out,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(['pdftoppm', '-jpeg', '-jpegopt', 'quality=85', '-singlefile',
                            '-scale-to', '2000', 'aten-paper-vs-current.pdf',
                            'aten-paper-vs-current-preview'], cwd=out,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        paper_points = [(float(r['paper_native_us']), float(r['paper_raised_us'])) for r in rows]
        shared_paper = [(float(r['paper_native_us']), float(r['paper_raised_us']))
                        for r in rows if r['status'].startswith('validated_')]
        comparison_summary = dict(
            original_figure=str(args.paper_figure.resolve()),
            original_figure_sha256=hashlib.sha256(original).hexdigest(),
            original_figure_unchanged=True, published_all_cases=band_counts(paper_points),
            published_same_timed_cases=band_counts(shared_paper), current=band_counts(points),
            limitation='Original figure axes are preserved: 50-us lower limits versus 10 us currently. '
                       'The paper mixes timing protocols; current uses five pairs. '
                       'Two current points use alternative finite inputs; seven cases lack validated timings.')
        (out / 'comparison-summary.json').write_text(json.dumps(comparison_summary, indent=2) + '\n')
    (out / 'README.md').write_text(
        '# ATen current results\n\n'
        'The scatter follows Figure 5: native CUDA time on the horizontal axis and '
        'LoopLibrarian resident GPU time on the vertical axis, in microseconds. '
        'The black line is parity; red dotted lines delimit a factor of two.\n\n'
        f"There are {len(points)} timed cases from the {len(rows)}-case paper cohort. "
        f"Green circles show {len(groups['paper'])} cases passing the original comparisons. "
        'Hollow orange diamonds show the separately recorded finite-input argmax/min '
        'measurements; their original NaN-input disagreements remain unresolved. '
        'Missing and failed cases have no fabricated coordinates and remain in '
        '`aten-current-data.csv` and `plot-summary.json`.\n\n'
        'All new points use five timing pairs. The original paper used mixed timing '
        'statistics; these points do not establish reproduction of every published '
        'speedup. The source report hash and plotting-script hash are recorded.\n')
    print(json.dumps({k: summary[k] for k in ('cohort', 'timed', 'original_inputs', 'alternative_inputs',
                                            'more_than_2x_faster', 'within_2x', 'more_than_2x_slower')}))


if __name__ == '__main__':
    main()
