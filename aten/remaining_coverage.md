# Remaining proof and comparison gaps

Use the latest run's `summary.json` and `gpu/report.json` for counts. Recognition,
successful linking, numerical agreement, and timing reproduction are separate
results; none implies the next.

## COO-to-CSR coverage

`aten_convert_coo_to_csr_cpu` and `aten_sparse_coo_to_csr_cpu` retain an
input-dependent `while` loop. Its cursor advances until the next row index is
at least the current output row. The generic region interpreter currently
handles counted `while` loops, but not this data-dependent recurrence.

The public PyTorch implementations in `TensorConversions.cpp` and
`SparseTensorUtils.cpp` fill gaps between adjacent row indices. Proving that
algorithm equivalent to the fixture's cursor scan requires a general loop
invariant and the sorted-row input condition. A function-name replacement,
copying the fixture into the library catalog, or assuming sortedness without
establishing it would not resolve this proof gap.

A scratch import of the public
[`convert_indices_from_coo_to_csr_cpu<int32_t,int32_t>` implementation](https://github.com/pytorch/pytorch/blob/d7af122d81a49b1fa7a31ba52bd57c026f092646/aten/src/ATen/native/TensorConversions.cpp)
passes the C/affine, Linalg, and debufferized execution comparisons. Its
symbolic interpretation now succeeds for both its affine and debufferized IR.
The generic interpreter resolves a previous-iteration input load when the
initial value agrees and the source array remains read-only. It handles both
SSA loop arguments and their scalar-memory representation. The application
still stops at a non-index `while` loop argument. This isolates the remaining
gap to symbolic loop reasoning, rather than the C
frontend or execution of the raised reference. The unproved reference is not
enabled in the production catalog.

The sorted-row condition is necessary, not just a proof optimization. A
two-element diagnostic with two rows gives fixture output `[0,0,2]` and
imported-reference output `[0,2,2]` for unsorted input `[1,0]`. All three sorted
two-element inputs agree. This small check is a counterexample to unconditional
equivalence, not a general proof on sorted inputs. Accepting the intended
sorted-input domain still requires explicit contract handling and the general
cursor-loop proof.

## Numerical criteria

The retained historical raised harness, `aten_pointwise_graph_silicon.py`, uses
`rtol * (1 + abs(reference))`, with `rtol = 0.002` for these FP32 cases.
This is also present at compiler commit `c58b6f11`. The retained native harness
`aten/harnesses/native_exact_region_bench.cpp` uses the same FP32 threshold;
its FP64 threshold is `1e-10 * (1 + abs(reference))`. Native checks reject
non-finite values; raised checks allow matching NaNs/infinities. Signed zeros
compare equal in those historical checks.

The newer strict diagnostic uses `1e-5 + 1e-4 * abs(reference)` and distinguishes
signed zeros. Its failures below are failures under that stricter diagnostic,
not evidence that the historical numerical criterion fails. Both criteria are
now computed on the same full output. `--timing-correctness publication`
explicitly selects the historical criterion for timing admission; strict
results, guards, input-corruption checks, and process status remain visible.
The older report keys named `original_paper_*` describe strict diagnostics on
the original input recipe; they do not name the historical tolerance.

All five previously untimed matched cases pass the historical criterion.
Five paired samples per case give these minimum times in microseconds
(LoopLibrarian / native CUDA): sparse norm 292.964 / 208.931, outer product
1038.159 / 373.349, transposed 3-D convolution weight gradient
3035.276 / 2748.200, ordinary 3-D convolution weight gradient
7535.982 / 7120.552, and BSR sparse addmv 666.217 / 4518.466.
These complete timing evidence for 113 matched cases, not 115 matches.

### Strict diagnostic results

The existing checks retain these distinctions:

- Sparse norm: raised and native CUDA agree; both differ from serial FP32 C.
  Both GPU results pass the FP64 diagnostic, while serial C does not.
- The two 3-D convolution weight-gradient cases: raised and native CUDA
  agree. Their serial-C comparison fails, and some outputs also exceed the
  existing per-element tolerance against FP64. See `numerical_diagnostics.py`.
- Outer product: raised output passes the C check; native CUDA differs on
  signed zero.
- BSR sparse addmv: both implementations individually pass the C comparison,
  but their pairwise check fails near zero. A tolerance relation is not
  transitive; two individual passes do not prove pairwise agreement.
- Argmax and argmin: the original NaN-containing recipe has different native
  CUDA behavior. The separately recorded finite native-harness recipe passes
  and is labeled explicitly when it supplies timings. The original failure
  remains visible.

The strict failures remain recorded. The historical criterion has a separate
pass count; alternative input recipes never replace the original checks.

## Performance

### Historical comparison audit

The current plot is not yet a strict reproduction of the historical timing
experiment. The following differences are confirmed by the historical native
harness, the current generated harness, and the recorded timing samples:

- `aten_unbind_copy_cpu`: the historical native harness calls
  `at::unbind_copy_out` with preallocated output row views. An earlier native
  adapter called `copy_` on a whole tensor. Native time changed from 3.982368 to
  0.267204 ms while raised time changes from 0.217664 to 0.237860 ms. Most of
  the apparent speedup change, 18.30 to 1.12, therefore comes from changing
  the baseline operation. The adapter now calls the historical API again:
  LoopLibrarian 236.357 us, native 4891.543 us, ratio 20.696.
- `aten_renorm_scale_factor`: the historical native path runs `add_out`,
  `reciprocal_out`, `mul_out`, `gt_out`, and `where_out`; an earlier adapter used
  `renorm_scale_factor_stub`. That changed native time from 1.381760 to
  0.251012 ms. The historical five-call path is restored: LoopLibrarian
  1152.532 us, native 1429.274 us. The raised runtime still regresses from
  the paper's 633.024 us.
- `aten_sampled_addmm_sparse_csr_cpu`: dimensions and the native API agree,
  but historical column indices use `p % columns`, whereas the current
  recipe uses `((i*17)+(i/(N/R))*13)%C`. Native time changes from 1.803584 to
  8.187084 ms; raised time remained close, 8.205024 versus 8.140748 ms.
  A fresh pair using the native column recipe for both implementations gives
  1824.928 / 1886.017 us. This changes the raised input pattern; it is not
  evidence of a 4.5-fold compiler improvement. Both recipes are retained and
  the selected timing recipe is labeled `native-paper`.

The historical calls are in `aten/harnesses/native_exact_region_bench.cpp`;
the current adapters are in `scripts/correctness/aten_fixture_batch.py` and
the generated per-case harnesses. Earlier measurements remain in their original
run directories. Existing speedup aggregates are descriptive counts,
not evidence that all included comparisons reproduce the paper.

### Generated-code performance

Whole imported references recovered single-call paths for saved-statistics
normalization and Kronecker products. The latter now runs close to the
published runtime at the paper shape. Removing initial copies before proven
complete library writes also reduces several layout-operation runtimes.

Some permutation cases still use a flat library copy plus descriptor-based
layout movement; the historical path used one cuTENSOR permutation. Recovering
that path needs an imported multidimensional copy reference and ABI support
that retain its layout through matching. The compiler must not infer a new
library computation solely from the benchmark name during lowering.

For example, `aten_im2col` changes from one cuTENSOR permutation to a strided
copy into a temporary followed by `cublasScopy_v2`; its recorded raised time
changes from 0.171648 to 0.742124 ms. The Q/K/V transformation changes from a
fused runtime wrapper to three `cudnnBackendExecute` calls and a separate
transform, with raised time increasing from 0.703360 to 4.697680 ms. These
are generated-code regressions in addition to the baseline differences above.

Normalization still incurs broadcast preparation, and the Q/K/V transform
needs a public reference and graph adapter that retain its multiple outputs.
The existing tensor-pointwise adapter accepts a single overwritten output.
Five timing pairs and the original historical statistic are recorded
separately; results with different statistics are not silently equated.

Sparse norm currently emits cuDNN multiplication followed by CUB reduction
and a scalar square root. The historical path used one `cublasSnrm2` call.
The new 292.964 us versus historical 108.960 us is therefore a change in the
generated library sequence, not just a correctness-tolerance issue. A whole
FP32 norm reference and equivalence proof are needed to recover that call.
