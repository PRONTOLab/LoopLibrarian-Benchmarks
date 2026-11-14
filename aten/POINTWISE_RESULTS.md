# Pointwise results and remaining gaps

## Latest completed matching checkpoint

The recorded union is **244/268**, with **24 candidates still lacking a
complete match**. This combines 195 other historical matches with 49 complete
cases from a fresh run of the 73-case cohort in
`results/runs/aten-pointwise-integer-captures/`. It is not a fresh run of all 268.
Every counted case passed proof verification and ABI lowering, with no residual
Linalg computation or loops. All 73 original fixture hashes are unchanged.
The previous 44 complete cases in this cohort remain complete; the new recoveries
are `aten_constant_pad_nd_cpu`, `aten_eye_cpu`, `aten_fft_conjugate_symmetry_cpu`, `aten_nested_to_mask_cpu`, `aten_triu_mask_cpu`.

The original helper cohort is **22/22**, including its twelve previously missing
cases. Those twelve were included in the original remaining-63 list, so the two
lists are not disjoint. **39/63** of that fixed list now have complete
matching records. `pointwise_inlining_cohort.csv` records the 73 unique cases;
`pointwise_remaining.csv` records the current unresolved cases. The original
`pointwise_remaining63.csv` remains unchanged.

Application and reference helpers are expanded before matching. The generic
region matcher compares complete typed regions, including recurrences, scalar
parameters, predicates and immutable tables. The Linalg matcher handles scalar
bodies and compositions. Matching does not depend on a helper or benchmark name.
`polygamma` uses the direct imported helper semantics; high-level ATen dispatch
for orders zero and one is outside that extracted fixture.

Recent additions include public Thrust typed arithmetic, copies, sequences,
counting-iterator division/remainder, integer comparisons and offset copy/negate
traversals. Trigamma imports the complete public CUDA Math.cuh implementation.
The compiler normalizes unsigned integer division/remainder across an index cast
only when the integer and target index widths agree. The proof engine retains
integer casts, overflow checks and signedness when solving captured literals.
C ABI discovery now accepts nested expressions in VLA declarations. Floating-point
operation order is preserved; there is no new floating reassociation rule.

The full compiler regression suite passes all **468 tests**. Added imports
pass exact CPU comparisons at the affine, Linalg and production tensor stages.
Public-header tests cover 108 iterator-window configurations, 70 coordinate and
predicate configurations plus 147 integer comparison pairs, alongside the earlier
sequence, typed-copy, shared-base-power and trigamma checks. The updated Thrust
runtime compiles for sm_87. **No GPU executions or timings were added by this
matching work.** The previously established 50 pointwise GPU cases remain a
separate result.

Remaining cases include special-function implementation differences, C++ aggregate
raising, masked or irregular layouts, scatter/accumulation, and fused updates.
A case with emitted calls but residual loops or Linalg operations remains partial.
The completion criterion has not been relaxed.

Reproduction commands and fixed cohorts are in [POINTWISE.md](POINTWISE.md).
Detailed counts and per-case evidence are recorded in
`results/runs/aten-pointwise-integer-captures/checkpoint.json`.

The earlier full strict scan of `pointwise_candidates.csv` had 154 complete matches,
19 partial matches, and 95 unmatched cases. All 268 fixtures reached raised IR
without a frontend or raising failure. These are candidate counts, not the
paper's 115-case denominator. Fifty basic pointwise cases outside that paper
cohort passed strict GPU checks and have five timing pairs each. Matching alone
does not establish correctness or performance for the other cases.

## Fixed 50-case matching batch

The subsequent `pointwise_batch50.csv` batch has **50/50 complete strict
matches**, up from 38/50. All 50 pass proof verification and ABI lowering, with
no remaining Linalg computation or loops. Original fixture hashes are unchanged.
The matching run took 311.8 seconds on the shared development host. It performed
no GPU correctness or timing runs, so this result does not add GPU passes or
performance evidence to the earlier results below.

The twelve newly complete cases are:

- `aten_clamp`, `aten_clamp_scalar_cpu`, `aten_clamp_cpu`;
- `aten_maximum`, `aten_minimum`, `aten_leaky_relu`;
- `aten_huber_elementwise`, `aten_huber_backward`;
- `aten_smooth_l1_elementwise`, `aten_smooth_l1_backward`;
- `aten_round`, `aten_round_decimals`.

Pure conditional prefixes now use the same generic splitting machinery as
arithmetic prefixes. Thirty-two reusable comparison/selection specializations
compose public ATen comparison and `where` scalar definitions; they use the
existing cuDNN graph emitter. One additional Thrust reference preserves exact
`roundf` primitive semantics. No application-name matcher, floating-point
identity, or relaxed numerical assumption was added. Source-fixture matching
remains distinct from native ATen API equivalence, particularly for rounding
and conditional edge cases.

The full 268-case inventory has not been rerun after these changes. The twelve
new matches are outside the paper's 115-case cohort. Reproduce this fixed batch
with the command in [POINTWISE.md](POINTWISE.md); its saved per-case results are
`results/runs/aten-pointwise-batch50/summary.csv` and the corresponding proofs
and ABI records under `matching/`.

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

## Earlier unmatched-case batch (superseded for 22 helper cases)

The fixed `pointwise_unmatched50.csv` cohort initially recorded **50/50 new
matches**, using strict imported-reference matching and successful ABI lowering.
None of these 50 was in the earlier 166 recorded complete matches. Every original
C source hash is unchanged, and all final variants have zero residual Linalg
computation and zero loops. All 50 are outside the paper's 115-case cohort.

The final matching run took 416.0 seconds (about seven minutes) with six workers
and a built compiler. Evidence is in `results/runs/aten-pointwise-unmatched50/`:
`summary.csv`, per-case proofs and ABI IR, the command receipt, and
`matching-audit.json`. No GPU correctness checks or timings were performed for
this cohort. The number of previously GPU-validated pointwise cases stays 50.

That result included 22 special-function traversals around opaque scalar helper
calls. Those 22 are withdrawn from the body-matching count: equality of helper
names does not prove their implementations. The other 28 are arithmetic,
conditional, integer or selection cases. The helper execution wrappers use
pinned public ATen templates through Thrust, but executable bindings alone do
not establish a valid computation-level match. Generic comparison and
selection compositions emit cuDNN graphs; integer operations and IEEE binary
math use typed Thrust wrappers. Shared normalization exposes pure scalar prefixes
and independent tensor outputs without adding application-name recognizers.

The final two gaps were a two-comparison predicate in `logit_backward` and the
shared `hypot` computation with two outputs in complex sign. The former matches
an imported interval-selection composition. The latter now emits one `hypot`,
two divisions and two selections, all proved against imported references.

The earlier recorded union was **217/268**: the earlier 166, these 50, and `add_clamp`,
which the fresh baseline found already complete due to the preceding fixes.
`add_clamp` was excluded from the new cohort and replaced by unmatched `where`.
The full 268-case inventory has not been rerun with this compiler, so this is a
union of recorded evidence, not a fresh full-inventory coverage measurement.
With the 22 opaque-helper cases excluded, 195 other recorded matches remain;
these are also historical evidence, not a fresh full-inventory measurement.

Before helper inlining, all 109 newly imported specialization variants passed
the production tensor reference execution check. The independent memref-Linalg diagnostic retains its
documented discrepancies; its outcome is separate from production tensor proof
eligibility. No preexisting reference import status changed. A special-function
match also successfully links into a Jetson executable; that executable was not
run on the GPU in this matching experiment. At that checkpoint, all 401 compiler regression tests
passed, including the pure-call congruence and two-output execution checks.

## Earlier full-body helper rerun (superseded)

The replacement `pointwise_inline_helpers.csv` run has **10/22 complete
matches**. Both applications and references import and inline the actual pinned
public ATen helper definitions before raising. All ten pass strict body matching,
stored-proof verification and ABI lowering, with no remaining Linalg computation,
loops or custom helper calls. Original fixture hashes are unchanged for all 22.

The ten complete cases are:

- Bessel J0 and Y0;
- modified Bessel I0, I1, K0 and K1;
- scaled modified Bessel K0 and K1;
- `ndtri` and spherical Bessel J0.

The other twelve are incomplete:

- J1 and Y1 retain a recursive J1 implementation call; `log_ndtr` retains a
  transitive helper call. The frontend rejects these before matching.
- Chebyshev T/U/V/W and Hermite H/He retain per-element loops and have no complete
  library match.
- Airy, Laguerre and Legendre have partial replacements but retain computation
  and loops. Airy emitted many small calls and took nearly the full case time
  limit; it is neither a complete match nor a performance result.

All ten imported tensor references pass exact CPU comparison. A separate
4,096-element Bessel J0 CPU check covers branch boundaries, signed zero,
nonfinite values, unchanged inputs and allocation guards. **No GPU correctness
or timing checks were run for these ten cases.** The previously GPU-validated
pointwise count remains 50; this rerun does not change the paper's 115-case cohort.

The common frontend now handles constant tables, promoted bounded counters,
compiler floating constants and early-return control flow. Shared recurrences
remain DAGs during scalar proof preparation. The region equation matcher
conservatively declines expressions whose tree expansion exceeds its resource
budget; the Linalg matcher can still prove those bodies. No operation-name
recognizer or wrapper-only fallback was added.

All 415 compiler tests and 59 benchmark-driver tests pass. A complete rebuild of
428 references leaves all 406 imports outside the affected helper group at
their previous statuses. The
22-case matching run took 596.4 seconds with four workers and a built compiler.
Evidence is in `results/runs/aten-pointwise-inline-helpers-complete/`, including
`summary.csv`, `inlining-audit.json`, per-case proof/ABI IR and the command receipt.

At that checkpoint, the recorded union was **205/268** (195 other historical matches plus these
10 freshly validated body matches). This remains a union of recorded evidence;
the full 268-case inventory has not been rerun on the final compiler.

## Remaining work

After withdrawing 22 opaque-helper records and recovering ten through body
matching, 63 candidates lack a complete body-match record.
The earlier full scan had 114 incomplete candidates (19 partial and 95 unmatched); the first
fixed batch added twelve, this batch recorded fifty (including the 22 withdrawn
helper matches), and one further case was found complete during baseline checking.
A fresh complete inventory scan is still needed to establish the effect of the generic fixes on other candidates.

- Resolve the twelve incomplete helper cases above and import remaining missing
  helper implementations; a helper name alone must not authorize a replacement.
- Add the missing imported operations and resolve equivalent arithmetic and
  conditional representations through the shared matcher.
- Establish index and layout equivalence for index-dependent candidates.
- Keep source-semantic differences visible: for example, C `roundf` and ATen
  round disagree on ties. A reference for one does not prove the other.
- Validate the remaining complete matches on GPU, adding actual native APIs
  where needed; then measure performance and investigate multi-call overhead.

## Evidence and reproduction

The local runs are under `results/runs/` (generated outputs, not source inputs):

- `aten-pointwise-inline-helpers-complete`: the superseding 22-case body-import
  rerun, with ten complete matches and no GPU execution.
- `aten-pointwise-unmatched50`: historical wrapper-level results and ABI checks;
  the 22 opaque-helper matches are superseded. No GPU execution or timings.
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
