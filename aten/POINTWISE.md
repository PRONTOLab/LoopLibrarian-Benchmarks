# Pointwise experiments

`pointwise_candidates.csv` records 268 candidates found by inspecting the saved
raised IR of the 598 unchanged C fixtures. It retains source hashes, IR hashes,
body categories, and ranks. Of these candidates, 54 overlap the paper's 115-case
cohort and 214 are outside it. This is an inventory to investigate, not a claim
that 268 kernels match, run correctly, or have native GPU baselines. Some entries
include indexing, copies, helper calls, or further layout constraints.

```
python3 aten/pointwise.py --output results/runs/pointwise --cases exp2 expm1 erfc
```

Omit `--cases` to scan the full inventory. Use `--stage match`, `build`, `run`,
or `report` to separate the stages. Matching needs a fresh output directory;
subsequent stages use that directory's saved proofs. Configure the compiler,
SDKs, native CUDA libtorch, and remote transport as for `aten/reproduce.py`.
Use the environment's Python with egglog installed.

ATen's Jiterator implementations of `erfc`, `exp2`, and `lgamma` also require
NVRTC on the target. Install the CUDA 12.6 NVRTC runtime, including its matching
`libnvrtc-builtins` library, and include their directory in `remote_library_path`.
The driver checks `libnvrtc.so.12` before these tests. NVIDIA's public
[CUDA redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.2.json)
provides the matching `cuda_nvrtc` 12.6.77 archive. The Jetson `linux-aarch64`
archive has SHA-256
`aafeadb7108b49ed6d432cc471262eafd9c585dfa3317b57815c59a727f872d4`.
This is an external runtime dependency; SDK binaries are not stored in this repo.

Both references and applications use the shared raising pipeline. The driver
enables generic rank normalization and pointwise-input splitting and requires
imported-reference EqSat evidence, successful ABI binding, and no remaining
Linalg computation or loops before building a case. Source function names and
the native comparison table do not authorize compiler replacements.

Matching defaults to `--numeric-model strict`. The optional `relaxed` model
enables the compiler's existing finite-arithmetic assumptions, including
reassociation and disregard of signed zero; exceptional intermediate values are
outside that proof model. It is not an IEEE equivalence proof. GPU comparisons
retain their strict checks in either mode, including exceptional values and
signed zeros. Reports record the matching model separately from the output-check
result. Use separate output directories for the two models; their evidence
cannot be combined into one coverage claim.

The runtime stage compares every output with the unchanged C fixture and actual
ATen CUDA, checks unchanged inputs and allocation guards, and exercises ordinary,
exceptional, and signed-zero inputs. It captures five alternating synchronized
resident timing pairs only after the strict checks pass. Outputs retain failed
comparisons. Native API expressions are saved in each generated bridge; where a
fixture represents an extracted stage, a composition of actual ATen APIs is
identified explicitly, including its full execution cost.

These experiments use original fixture dimensions. Their timings are separate
from Figure 5's paper-shape experiment; they must not be presented as a numerical
reproduction of that figure. `summary.json` separates candidates, complete
matches, strict GPU passes, and timings. A missing native API prevents execution
validation and is not counted as a pass.

To combine separately executed subsets, use `--stage report` with
`--extra-gpu-results path/to/other-run/gpu/gpu-results.json`. The report rejects
duplicate cases and verifies matching source, compiler, ABI and specification
inputs before counting external evidence. Larger-shape results remain separate.

For larger measurements, `pointwise_shapes.json` retains the archived resident
dimensions for 166 inventory entries, with hashes of the archive and each source
record. Supply it explicitly, selecting cases present in that manifest:

```
python3 aten/pointwise.py --shape-specs aten/pointwise_shapes.json --cases exp2 expm1 erfc --output results/runs/pointwise-resident
```

This materializes separate C copies with the recorded dimensions, checks them
again, and runs five timing pairs at those dimensions. These records describe the
archived resident benchmark; their presence does not imply inclusion in Figure 5
or reproduction of its original timing boundary.

Allow approximately 2–10 minutes for a selected case with a built compiler,
20–60 minutes for the full matching inventory, and additional compilation and GPU
time for executable cases. These are planning estimates, dependent on shared-host
load and build-cache state; command receipts record actual elapsed time.

Known source-semantic distinctions remain explicit. For example, the saved
`aten_round` fixture calls C `roundf` (ties away from zero), while ATen's native
round uses ties to even. Importing one primitive does not establish equivalence
to the other. The driver does not modify fixtures or relax checks to hide this.

## Fixed 50-case matching batch

`pointwise_batch50.csv` records 50 historical graph candidates, their unchanged
C source hashes, archived call/IR hashes, and the earlier matching status. The
batch contains 38 previously complete matches and 12 previously incomplete
ones. It is fixed before the new run; cases are not selected from its passing
results.

```
python3 aten/pointwise.py --stage match --checklist aten/pointwise_batch50.csv --output results/runs/aten-pointwise-batch50 --jobs 6
```

This performs fresh C raising, strict imported-reference matching, proof
verification and ABI lowering for all 50. It does not build GPU executables or
run GPU correctness/performance experiments. A complete match requires no
remaining Linalg computation or loops, in addition to emitted calls and a valid
ABI. `summary.json` separates baseline and newly complete matches from GPU
validation and timing counts. Use a fresh output directory for a repeat run.

Comparison/selection compositions are imported from the public ATen scalar
comparison and `where` definitions. Generic prefix splitting exposes pure
conditional subexpressions; it preserves branch order rather than assuming
that nested conditionals are equivalent to min/max for all bounds. These
references use the existing cuDNN graph emitter. The rounding reference uses
actual `roundf` semantics, which remain distinct from native ATen ties-to-even
rounding. Matching the saved C fixture does not establish native-API equivalence.

## Previously unmatched 50-case batch

`pointwise_unmatched50.csv` is a separate, fixed cohort whose 50 entries were
incomplete in the recorded inventory and were not among the earlier batch's
completed matches. A fresh baseline identified `add_clamp` as already complete;
it was excluded before fixing this cohort and replaced by the unmatched `where`
fixture. The cohort contains 22 special-function traversals and 28 arithmetic,
comparison, integer, or selection cases. Original C sources and hashes are kept.

```
python3 aten/pointwise.py --stage match --checklist aten/pointwise_unmatched50.csv --output results/runs/aten-pointwise-unmatched50 --jobs 6
```

Special-function fixtures and references import and inline their pinned public
ATen scalar definitions before raising. The shared source-dependency configuration
is `generic_solver/lib_v3/support/aten_scalar_frontend.json` in the compiler.
Remaining helper calls, including recursive implementation calls, fail this step.
The earlier wrapper-only results do not establish matching of those computations.
Public comparison/where compositions and integer transform references cover the
other primitive operations. These are library specifications, not application-specific
recognizers.

To check just the 22 affected special-function cases, use the same command with
`--checklist aten/pointwise_inline_helpers.csv` and a fresh output directory.
Each case saves the generated C++ input, the expanded frontend IR, the raised IR,
and any matching/ABI evidence. Original fixture files and hashes remain unchanged.

This command checks raising, imported-reference matching, and ABI emission. It
does not run GPU correctness or timing experiments. The CPU reference-import
checks and CUDA wrapper compilation are separate checks; neither establishes
GPU output correctness for these 50 fixtures.

## Remaining-case cohorts

The original remaining-63 cohort is saved in `pointwise_remaining63.csv`.
It includes the twelve formerly incomplete cases from the 22-helper cohort;
there are 51 other cases, not 63 additional cases. The six further typed
reference imports have a fixed cohort in `pointwise_reference_extensions.csv`.
For example:

```
python3 aten/pointwise.py --stage match --checklist aten/pointwise_reference_extensions.csv --output results/runs/aten-pointwise-reference-extensions --jobs 6
```

The combined helper/remaining cohort contains 73 unique cases. Run it with
`--checklist aten/pointwise_inlining_cohort.csv`; run only the current unresolved
cases with `--checklist aten/pointwise_remaining.csv`. The latter list is a
checkpoint, while `pointwise_remaining63.csv` preserves the original cohort.

The latest complete-cohort check also exercises imported counting-iterator
division and remainder, integer comparisons, and offset copy/negate operations:

```
python3 aten/pointwise.py --stage match --checklist aten/pointwise_inlining_cohort.csv --output results/runs/aten-pointwise-integer-captures --jobs 6
```

Use another fresh output directory when repeating the command. These references
describe public Thrust operations and retain their integer types and conversions.
They use the same raising and proof pipeline as the other imports. This command
checks matching and ABI lowering; it does not execute CUDA or measure performance.

Smaller fixed follow-ups are `pointwise_typed_primitives.csv` (typed arithmetic),
`pointwise_sequence.csv` (sequence generation), and `pointwise_typed_copy.csv`
(integer copies and logspace). For example:

```
python3 aten/pointwise.py --stage match --checklist aten/pointwise_typed_copy.csv --output results/runs/aten-pointwise-index-copy --jobs 3
```

The default operand rank limit is one. `--ranks 1 2` explicitly tests both
limits through the same normalization and matching pipeline. The report still
requires a complete computation and successful ABI emission; raising or a
partial initializer match does not count as completion.

## Extending the experiment

Add an unchanged extracted fixture to `aten/inputs/`, inspect its raised IR, and
record its source hash and candidate category in `pointwise_candidates.csv`.
Run `--stage match` first. If a reference is missing, add its public source
provenance and parameterized C import to the compiler's `lib_v3` catalog and run
reference import validation. Existing graph adapters emit supported scalar
instructions directly from the raised reference. A new fixed-ABI vendor wrapper
is needed only where the existing adapters cannot express the library API.

Add the corresponding real ATen CUDA call to the benchmark's native baseline
table, then run `build` and `run`. This table is deliberately separate from
compiler matching. Keep original-C, native-CUDA and raised comparisons separate;
record an unavailable native equivalent or a numerical failure explicitly.

See [the measured pointwise results and remaining gaps](POINTWISE_RESULTS.md)
for the completed scan. The inventory and executable validation counts are
reported separately.
