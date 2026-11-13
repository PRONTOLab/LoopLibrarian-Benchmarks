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
symbolic interpretation stops when a loop-carried value is needed as an array
index; the application stops at a non-index `while` loop argument. This
isolates the remaining gap to symbolic loop reasoning, rather than the C
frontend or execution of the raised reference. The unproved reference is not
enabled in the production catalog.

## Original numerical comparisons

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

The report does not widen tolerances or remove exceptional inputs to increase
the pass count.

## Performance

### Historical comparison audit

The current plot is not yet a strict reproduction of the historical timing
experiment. The following differences are confirmed by the historical native
harness, the current generated harness, and the recorded timing samples:

- `aten_unbind_copy_cpu`: the historical native harness calls
  `at::unbind_copy_out` with preallocated output row views. The current native
  adapter calls `copy_` on a whole tensor. Native time changes from 3.982368 to
  0.267204 ms while raised time changes from 0.217664 to 0.237860 ms. Most of
  the apparent speedup change, 18.30 to 1.12, therefore comes from changing
  the baseline operation.
- `aten_renorm_scale_factor`: the historical native path runs `add_out`,
  `reciprocal_out`, `mul_out`, `gt_out`, and `where_out`; the current path uses
  `renorm_scale_factor_stub`. Native time changes from 1.381760 to 0.251012 ms;
  raised time also regresses from 0.633024 to 1.149012 ms.
- `aten_sampled_addmm_sparse_csr_cpu`: dimensions and the native API agree,
  but historical column indices use `p % columns`, whereas the current
  recipe uses `((i*17)+(i/(N/R))*13)%C`. Native time changes from 1.803584 to
  8.187084 ms; raised time remains close, 8.205024 versus 8.140748 ms. A
  controlled rerun is required to isolate the effect of the index pattern.

The historical calls are in `aten/harnesses/native_exact_region_bench.cpp`;
the current adapters are in `scripts/correctness/aten_fixture_batch.py` and
the generated per-case harnesses. Restore the historical API scope and input
recipes for the reproduction comparison, retaining changed-baseline
measurements separately. Existing speedup aggregates are descriptive counts,
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
