# Pointwise results and remaining gaps

The fresh strict scan of `pointwise_candidates.csv` has 154 complete matches,
19 partial matches, and 95 unmatched cases. All 268 fixtures reached raised IR
without a frontend or raising failure. These are candidate counts, not the
paper's 115-case denominator. Fifty basic pointwise cases outside that paper
cohort passed strict GPU checks and have five timing pairs each. Matching alone
does not establish correctness or performance for the other cases.

## Added reference imports

Eleven math operations previously lacked a complete match: `exp2`, `expm1`,
`log2`, `log10`, `erfc`, `lgamma`, `hypot`, `fmod`, `fmax`, `fmin`, and `nextafter`.
They now match imported public scalar references and lower to actual Thrust
transforms. MSE elementwise and backward already matched as two smaller calls;
the two new MSE reference imports permit a single cuDNN graph call instead.
Original ATen fixture bodies were not edited.

Reference and residual execution now use upstream Math-to-libm patterns before
LLVM math lowering. This retains accurate `expm1` and `log1p` behavior near zero
instead of expanding them as subtraction/addition around `exp` and `log`.
Executable tests cover small FP32 values and signed zero.

## Larger measurements

Eleven of the thirteen new specializations have archived resident dimensions
of 4,194,304 elements. All eleven passed strict comparisons with both original
C and actual ATen CUDA. Ten were faster than ATen; elementwise MSE was about
3.5% slower. The geometric mean of native-time / raised-time was 1.15.

These are archived resident dimensions, separate from Figure 5. Five alternating
timing pairs were collected. The MSE-backward baseline is explicitly two actual
ATen calls (subtraction and in-place multiplication), because the extracted
fixture accepts an arbitrary normalization parameter. It is not labeled as a
single native MSE-backward kernel.

The GELU variants pass output checks but still emit nine primitive calls. At
the same large dimensions they achieve only about 0.10 times native ATen speed.
A separate relaxed matching experiment did not recover the whole reference;
it is not included in the strict inventory evidence. Expression grouping and
the representation of captured constant conversions need further work before
claiming a whole-reference GELU performance result.

## Remaining work

The 114 incomplete candidates consist of 19 partial and 95 unmatched cases.
The latter contain 38 scalar-helper bodies, 38 arithmetic bodies, 15
index-dependent bodies, and four copy/fill or mixed bodies. These are IR
categories, not a claim that every case in a category has one identical cause.

- Import and raise public implementations of opaque scalar helpers, including
  special functions; a helper name alone must not authorize a replacement.
- Add the missing imported operations and resolve equivalent arithmetic and
  conditional representations through the shared matcher.
- Establish index and layout equivalence for index-dependent candidates.
- Keep source-semantic differences visible: for example, C `roundf` and ATen
  round disagree on ties. A reference for one does not prove the other.
- Validate the remaining complete matches on GPU, adding actual native APIs
  where needed; then measure performance and investigate multi-call overhead.

## Evidence and reproduction

The local runs are under `results/runs/` (generated outputs, not source inputs):

- `aten-pointwise-full`: full matching scan, combined 50-case validation,
  `summary.csv`, and the per-case `remaining.csv`.
- `aten-pointwise-math`: correctness and timings for the thirteen imports at
  original fixture dimensions.
- `aten-pointwise-previous-catalog`: diagnostic with those imports disabled;
  eleven cases fail to match and both MSE cases retain their two-call matches.
- `aten-pointwise-math-resident`: the eleven larger measurements.
- `aten-pointwise-gelu-relaxed`: separately labeled unsuccessful GELU performance
  experiment, with passing output checks and nine calls retained.

Use `aten/pointwise.py` as described in [POINTWISE.md](POINTWISE.md) to regenerate
the scan and GPU results. Saved source hashes, matching proofs, ABI checks,
compiler receipts, and numerical-model settings accompany the outputs. Fresh
results may vary with compiler changes and shared-machine load.
