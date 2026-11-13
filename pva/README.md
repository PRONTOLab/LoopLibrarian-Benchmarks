# PVA correctness

Run fresh public C fixtures through the shared raising pipeline, prove them
against imported public OpenCV references, emit typed PVA calls, and compare
complete outputs on the target. No PVA timing claim is required.

```
python3 reproduce.py pva --output results/runs/pva
```

The driver currently covers dilation for four integer pixel types, the
8-bit-input histogram with 32-bit signed/unsigned output arrays, and 8-bit
3-by-3 Gaussian filtering, and U8 box filtering with an explicit quantization
contract. Histogram and image filters use wrappers fixing the
benchmark dimensions to 192 by 256; their original C
bodies are unchanged. This is fixed-shape coverage, not a claim to support every
dynamic image size. The signless MLIR counter type permits either validated
32-bit PVA output route because all counts are nonnegative and below INT_MAX;
the emitted route is recorded in each build receipt.

Each case retains raised IR, an imported-specification proof, the linked
executable hash, complete correctness output, and the observed vendor call.
The histogram harness checks every bin. Image filters check the complete output,
including preservation of the untouched border. A summary lists the operator
families that the fresh flow does not yet reproduce: BilateralFilter and
HistogramEqualization. The other Gaussian pixel types
and 16-bit-input histogram routes also remain outside this driver.

Gaussian imports OpenCV's public fixed-point implementation. Generic integer
range checks preserve casts, shifts, overflow behavior, and rounding while
proving its equivalence to the unchanged C stencil. The shared ABI configuration
evaluator derives sigma from the imported reference's coefficient shifts,
rounds the API parameters to FP32, and checks the resulting quantized weights.
The generated call passes sigma 0.8493217825889587 on both axes. The runtime
wrapper does not contain a fixed sigma. This supports imported compile-time
specializations; arbitrary application coefficient arrays are not yet searched.

BoxFilter also requires an exact source-to-reference match. Its separate PVA
backend contract bounds quantization error by 3 over U8 pixels, conditional on
the documented coefficients and rounding model. The driver explicitly enables
that specification's budget; normal compiler matching keeps approximate
backends disabled. The bound comes from rational interval arithmetic, rather
than from the observed test errors. Proof receipts retain the configuration,
model assumptions, calculated bound and accepted budget. Changing those inputs
invalidates stored match evidence.

Gaussian and BoxFilter execute five input patterns: original initialization,
random, impulse, constant and a quantization counterexample. The harness checks
all pixels, untouched input, exact border preservation and buffer guards. On
the documented target, Gaussian is exact for all five; BoxFilter's observed
maximum errors are respectively 1, 2, 1, 0 and 3. These observations validate
these inputs on this backend and do not prove the closed implementation follows
the model for every input. Results distinguish `exact` from `bounded-quantization`.

Install the external PVA SDK and operator binaries separately. Configure
`PVA_OPERATOR_INCLUDE`, `NVCV_INCLUDE`, `CUPVA_SDK_ROOT`, and `PVA_LIB_STAGE` in
the local configuration's `environment` mapping. They point to installed
headers/libraries; the repository does not redistribute SDK source or
headers. Set the SSH transport and `remote_library_path` in `config.json` as
described in the top-level README.

Allow approximately 5–10 minutes to build these eight cases on the documented
host, plus 1–3 minutes for deployment and correctness checks. SDK/toolchain
installation is separate. To resume a completed build, use `--stage run` with
the same output directory; use `--cases morphology_u8` for a small smoke test.

BoxFilter and the remaining two operator families are not exact substitutions for the
unchanged C fixtures. A separate diagnostic calls their real PVA APIs and
compares full outputs directly with those fixtures:

```
python3 pva/vendor_compatibility.py --output results/runs/pva-vendor-compatibility
```

This takes about 1–2 minutes with the installed toolchain and SDK. It does not
run the compiler matcher, produce timings, or count as reproduced coverage.
The output records every mismatch, maximum absolute error, input preservation,
and guard bytes for the original filter initialization, random pixels, an
impulse image, a constant image, and a quantization counterexample. The last
input places 255 at the four kernel positions whose Q8 coefficients are
incremented by the documented BoxFilter adjustment. The original C function
still supplies the expected output. A zero process exit means collection
completed; inspect `summary.json` for numerical agreement.

Observed differences on the documented target are:

- BoxFilter: maximum error 1 for the original initialization, 2 for random
  pixels, and 3 for the quantization counterexample. PVA uses quantized,
  unequal integer coefficients; the C fixture
  computes an exact rounded average. See the
  [public BoxFilter description](https://docs.nvidia.com/pva/solutions/0.4.0/impl/operator/boxfilter.html).
- BilateralFilter: maximum error 4 for the original initialization and 18 for
  random pixels. The C fixture visits all nine square-neighborhood pixels;
  the public library references use a circular neighborhood, and PVA also
  quantizes weights. See the
  [public BilateralFilter description](https://docs.nvidia.com/pva/solutions/0.5.0/impl/operator/bilateralfilter.html).
- HistogramEqualization: maximum error 1 for the original initialization and
  2 for random pixels. For an image filled with 73, the fixture produces zero
  while the installed PVA API preserves 73. The library also uses fixed-point
  normalization. See the
  [public algorithm description](https://docs.nvidia.com/pva/solutions/0.3.0/impl/operator/histogramequalization.html).

The OpenCV box reference now uses the explicit contract described above.
Bilateral's neighborhood difference and histogram equalization's constant-image
behavior remain outside that contract. They are still excluded from production
matching; changing a comparison tolerance alone cannot authorize a replacement.
