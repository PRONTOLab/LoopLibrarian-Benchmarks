# Public PVA fixtures

This directory retains synthetic fixture inputs, harnesses, and archived output
checks. Calendar stamps and internal machine details have been removed. These
records do not establish a fresh compiler-to-PVA reproduction.

The public harness builder is `scripts/correctness/build_pva_silicon_tests.sh`.
It requires separately installed, compatible SDK components configured through
`PVA_OPERATOR_INCLUDE`, `NVCV_INCLUDE`, `CUPVA_SDK_ROOT`, and `PVA_LIB_STAGE`.
The SDK is not redistributed. Public SDK acquisition and fresh installation
validation remain part of completing the PVA artifact workflow.

No PVA runtime speedup is a paper reproduction target.
