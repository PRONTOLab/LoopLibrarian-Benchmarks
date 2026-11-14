# PVA correctness — Table 7

Raise the public C fixtures, match them against imported OpenCV references,
emit typed PVA calls, and compare complete outputs on the target. This experiment
does not measure PVA performance.

```
python3 reproduce.py pva --output results/runs/pva
```

Install the PVA SDK and operator libraries separately. In `config.json`'s
`environment` mapping, set `PVA_OPERATOR_INCLUDE`, `NVCV_INCLUDE`,
`CUPVA_SDK_ROOT`, and `PVA_LIB_STAGE` to their installed paths. Configure target
execution and shared libraries using the [setup guide](../docs/reproduction.md).
SDK sources and headers are not redistributed here.

Allow 5–10 minutes to build and 1–3 minutes to run. The eight cases cover dilation
for four integer pixel types, an 8-bit-input histogram with signed/unsigned
32-bit outputs, 8-bit 3×3 Gaussian filtering, and 8-bit 3×3 box filtering.
Histograms and filters use fixed 192×256 inputs. Select a case with
`--cases morphology_u8`, or resume a built experiment with `--stage run`.

`summary.json` records each result. Per-case files retain raised IR, matching
proof, emitted API configuration, and full-output checks. Gaussian and box
filters exercise original, random, impulse, constant, and quantization-edge
inputs, including border preservation, unchanged inputs, and buffer guards.

Gaussian checks exact equality; its API parameters are derived from the imported
reference's coefficients. BoxFilter uses an explicitly enabled backend error
bound of 3 over 8-bit pixels under the recorded coefficient and rounding model.
The build records retain that model, the calculated bound, and accepted budget;
results distinguish exact equality from bounded quantization error.

BilateralFilter and HistogramEqualization remain unsupported: their library
semantics differ from the C fixtures. Other Gaussian pixel types and 16-bit-input
histograms are also outside this driver. The eight cases therefore do not
establish reproduction of every entry in Table 7.
