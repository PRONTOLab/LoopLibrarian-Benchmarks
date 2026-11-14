# ATen

This experiment measures library matching across 598 extracted/transliterated
ATen C kernels and compares LoopLibrarian with ATen CUDA on the 115 cases used
in Figure 5. Sources are in `inputs/`; extraction records are in `metadata/`.

## Requirements

Complete the [compiler setup](../docs/reproduction.md). GPU runs require Orin
and target-compatible CUDA, cuDNN, cuTENSOR, and libtorch. Set
`ATEN_NATIVE_TORCH_ROOT`, `ATEN_NATIVE_CUDA_ROOT`, and `POLYGEIST_CUTENSOR_ROOT`
in `config.json`'s `environment` mapping, and configure the execution target.
Plots require `pdflatex` with PGFPlots and `standalone`, plus `pdftoppm`.

## Choose the experiment

For Figure 5, use the 115 cases and dimensions recorded in `paper_cases.csv`
and `paper_shapes.json`:

```
python3 reproduce.py aten --cohort paper --timing-correctness publication --output results/runs/aten-paper
```

This compiles the cases, compares outputs, collects five alternating synchronized
LoopLibrarian/ATen timing pairs after warmup, and generates the plot. Allow
1–3 hours, or 2–10 minutes for one case selected with `--cases aten_mm`.
The matching options are in `matching_options.json`; published timings are in
`paper_timings.csv`.

For matching across all 598 inputs, using their source dimensions:

```
python3 reproduce.py aten --cohort all --stage match --output results/runs/aten-all
```

Omit `--stage match` to also build and run cases with complete matches and a
native CUDA comparison. Native comparison code exists for 170 inputs; the report
identifies cases that cannot yet be built or executed. Allow an estimated
1–3 hours for matching and another 1–4 hours for supported GPU runs. These
full-set timings use different dimensions from Figure 5.

## Outputs and checks

- `matching/`: per-case raised MLIR, library matches, and proof records.
- `gpu/`: build records, full-output comparisons, and raw timing samples.
- `gpu/report.csv`: one row per paper case, including failed and unmatched cases.
- `plots/aten-current.pdf` and `.png`: Figure 5 plot; the accompanying CSV includes missing results.
- `summary.json`: aggregate counts. Full-set runs also produce `summary.csv`.

A completed run does not imply that every case passed. Only validated timings
appear in the plot. COO-to-CSR conversion still requires unsupported loop proofs.

The paper command uses the FP32 error tolerance `0.002 * (1 + abs(reference))`.
The runner also records a stricter check, `1e-5 + 1e-4 * abs(reference)`, which
additionally distinguishes signed zeros. Omitting `--timing-correctness publication`
requires the stricter check before timing. Both modes check complete outputs,
input preservation, and buffer guards. Some reductions and convolution gradients
agree with ATen CUDA while failing the stricter comparison with serial C.

New timing results use the minimum of five samples per implementation. The paper
used five samples for 58 cases and twenty for 57 cases; the per-case protocol is
recorded in `paper_timings.csv`. Argmax/argmin timings use a separately checked
finite-input variant because ATen CUDA differs on the original NaN inputs.
Sampled sparse addmm uses the same column indices for both implementations;
its paper measurements used different indices. The reports label these input
and statistic differences, which limit direct comparisons with those paper points.

## Resume or plot

Use the same output directory to continue after matching or to redraw a plot:

```
python3 reproduce.py aten --cohort all --stage build --output results/runs/aten-all
python3 reproduce.py aten --cohort all --stage run --output results/runs/aten-all
python3 reproduce.py aten --cohort paper --stage plot --output results/runs/aten-paper
```

`--stage report` refreshes the summary. `--stage plot` uses saved timings and
takes seconds; add `--paper-figure path/to/figure.pdf` for a side-by-side comparison.
`--jobs` controls compilation parallelism; GPU timing runs are serial.
For additional pointwise tests, see [POINTWISE.md](POINTWISE.md).
