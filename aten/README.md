# ATen paper-cohort reproduction

The timing cohort contains 115 cases. The 598 extracted/transliterated source
fixtures are ordinary files in `aten/inputs/`; the cohort, paper dimensions,
matching configuration, and original timing records are in `paper_cases.csv`,
`paper_shapes.json`, `matching_options.json`, and `paper_timings.csv`.
Extraction provenance is in `aten/metadata/`. Fresh reproduction does not
require the evidence archive.

```
python3 reproduce.py aten --output results/runs/aten-paper
```

This command materializes separate paper-shape copies, raises them, proves
matches against imported `lib_v3` references, builds complete matches, checks
full outputs against the original C and real ATen CUDA, and runs five warmups
and five alternating synchronized timing pairs. The original fixture bodies
remain unchanged. Per-case configuration chooses generic raising and matching
options; it does not authorize a replacement without a specification proof.

Imported tensor references can retain broadcast dimensions instead of expanding
every input to the output size. For example, the public PyTorch saved-statistics
normalization reference supplies the arithmetic and broadcast maps for a cuDNN
graph; the ABI adapter only creates descriptors and marshals buffers. The
Kronecker-product reference exposes PyTorch's broadcast views and output reshape
and binds to cuTENSOR. Neither path recognizes application function names.

For a short test, add `--cases aten_pixel_shuffle`. Build and run can be
separated with `--stage build` and `--stage run`, using the same output directory.
Configure the target's SSH transport and library paths in `config.json`.
`ATEN_NATIVE_TORCH_ROOT` points to the target-compatible libtorch headers and
libraries on the build host; `ATEN_NATIVE_CUDA_ROOT` points to the AArch64 CUDA
installation. `POLYGEIST_CUTENSOR_ROOT` is also required by cases that prove a
cuTENSOR replacement. This setting is used for both runtime compilation and
linking the final ATen comparison executable. SDK setup is separate from the
experiment.

The output contains `matching/` with per-case IR, proofs and ABI checks, and
`gpu/` with executable hashes, correctness logs, every timing sample and a
combined `report.json`. The summary keeps these separate:

- complete imported-specification matches;
- successful executable builds;
- raised-versus-original-C correctness;
- native-versus-original-C correctness and raised/native agreement;
- validated timing pairs and both latency and native-relative speedup comparisons.

`gpu/report.csv` provides one row per selected paper case, including unmatched
cases. It records the emitted specifications and vendor calls, each correctness
criterion, timing input/statistic, and current versus published runtimes. Blank
cells mean unavailable evidence, rather than a successful check or zero time.

Regenerate the Figure 5-style scatter from a recorded report:

```
python3 aten/plot_results.py --report results/runs/aten-paper/gpu/report.json --output results/runs/aten-paper/plots
```

This requires `pdflatex` with PGFPlots and the `standalone` class, plus
`pdftoppm`. It writes a PDF, PNG, plot provenance, and a CSV retaining every
cohort case. Original-input passes and the separately validated finite-input
argmax/min timings use different markers. Failed or unmatched cases remain in
the CSV without invented timing coordinates. Rendering uses the saved samples
and takes a few seconds.

Add `--paper-figure PATH_TO_ARCHIVED_FIGURE.pdf` to produce an additional
side-by-side PDF/PNG with the original paper figure embedded unchanged. Its
summary compares both the full historical cohort and the cases with current
validated timings, and records the original figure hash. The original figure's
axis limits are retained, so consult the numeric comparison as well as the image.

Unmatched cases and failed comparisons remain in the denominator. A successful
driver exit means the experiment completed, not that every published result
was reproduced. Timing comparisons explicitly flag a different historical
statistic. Argmax/min retain the original NaN-containing recipe and its native
CUDA disagreement; their separately recorded finite native-harness recipe is
also checked and is identified whenever it supplies a timing measurement.
No tolerance is changed to hide a failed comparison.

Strict comparability with the paper is not established for every timing pair.
The current native harness uses different API scopes for `unbind_copy` and
`renorm_scale_factor`, and sampled sparse addmm uses a different column-index
recipe. Passing output comparisons does not resolve these timing differences.
See [the comparison audit and remaining work](remaining_coverage.md) before
interpreting the scatter or speedup counts as a reproduction of the paper.

Allow roughly 1–3 hours for the full fresh match/build/check/timing campaign on
the documented machines, depending on runtime-object cache state and host
load. A single case normally takes a few minutes. These are wall-time estimates,
not kernel runtimes. Compilation-cost experiments disable application compiler
caches and are measured separately.

Historical checkpoints, including their older counts, statistics and failure
diagnoses, remain under `results/archive/issues/`. Runtime-only relink probes
are diagnostics; a fresh source-to-executable run is needed before treating an
optimization as artifact evidence.

For reduction-order diagnostics, compare a recorded executable's computation
with the same paper-shape C body evaluated in FP64:

```
python3 aten/numerical_diagnostics.py --gpu-results results/runs/aten-paper/gpu --output results/runs/aten-numerics
```

This runs sparse norm and the two 3-D convolution weight-gradient cases. It
widens the initialized FP32 values exactly, evaluates the unchanged arithmetic
body in FP64, and reports the existing absolute/relative tolerance against
that reference. It retains the base-build and object hashes. Allow 1–3 minutes.
This is a diagnostic relink, with no timing measurements; it never overwrites
the original paper comparisons or counts as a new compiler reproduction.

The recorded diagnostic found that sparse norm's serial FP32 accumulation
fails the FP64 comparison while both GPU paths pass. The convolution cases
are more subtle: both GPU paths agree, but 17/3456 outputs in transposed
convolution and 92/3456 outputs in ordinary convolution still exceed the
existing per-element tolerance against FP64. Their original C references also
fail for 94/3456 and 85/3456 outputs respectively. These results establish
native-GPU agreement, not a clean pass of the original numerical criterion.
