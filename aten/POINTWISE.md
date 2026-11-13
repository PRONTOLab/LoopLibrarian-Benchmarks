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
