# Repository migration validation

Validated on 2026-11-12. Compiler commit: `2ac275444fbdf2b60a39cfb133aba2514c33b920`.
The scripts use the same raising, imported-specification matching, and ABI
lowering as the compiler branch. No benchmark computation was changed.

- Preserved 14,719 transferred paths; benchmark sources and archived
  results compare byte-for-byte with Polygeist `f824f9b5`. The migration map
  and verification JSON identify intentionally ported drivers and manifests.
- Re-imported all 260 specification entries. Every entry retained its prior
  validation status; this is not a claim that all 260 pass every semantic stage.
- Verified 239 upstream source hash references.
- Compiler suite after separating benchmark files: 313 tests ran successfully.
  Its Llama fixture class initially skipped because its input moved; the six
  checks in that class were then moved to this repository.
- All 28 benchmark tests pass, including those six Llama checks and the ATen
  fixture, ABI, shape, and public-pooling checks.
- All ten reproduction entry points pass their argument/help checks.
- Fresh PolyBench GEMM MINI host build emits `openblas_dgemm_nn`; native and
  raised executions pass full-output comparison. These are migration checks,
  not replacements for the Orin LARGE paper timing campaign.
- Fresh ATen `aten_mm` at its recorded paper shape raises, matches, and passes
  ABI lowering. GPU correctness/timing was not rerun for this layout change.
- The generic external iteration driver builds and executes a control program.
- The inventory entry point reports 260 current specifications versus the
  paper's 257; it preserves that difference instead of forcing the count.

The PolyBench CPU reproduction driver preserves the previously validated
post-compilation entry rename. The optional compiler mode that renames the
entry before C lowering still leaves unresolved strided-metadata operations
for GEMM in our probe; that mode is not used by the reproduction driver.

Full GPU timing campaigns were not repeated for this repository move. Existing
scientific limitations (including ATen's incomplete 115-case cohort) remain
listed in the suite documentation.
