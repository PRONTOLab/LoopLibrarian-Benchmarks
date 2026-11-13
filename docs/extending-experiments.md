# Reusing and extending the experiments

This guide supports our application for the **Artifacts Evaluated — Reusable**
badge. The badge is a target pending evaluation, not an award already received.
[CGO's criteria](https://2027.cgo.org/track/cgo-2027-artifact-evaluation) emphasize
documentation and structure that enable customization beyond the paper's cases.

Run commands from the repository root after completing the
[setup instructions](../README.md#getting-started) and configuring `config.json`.
Use the pinned compiler in `third_party/Polygeist`. Each experiment should have
a fresh output directory under `results/runs/`; keep the paper inputs and
archived results unchanged.

## Change a case or dataset without changing the compiler

For example, inspect the command for a single GEMM using PolyBench's MINI data:

```
/usr/bin/python3 reproduce.py polybench_cpu --stage plan --cases gemm --size MINI --output results/runs/reuse-gemm-mini
```

`--stage plan` prints the command without building or running it. Replace
`--stage plan` with `--stage all` to compile and check the native and raised
programs, then collect timings on the configured Orin CPU. To change the data
size, use `--size SMALL` and a new output directory. The supported sizes are
`MINI`, `SMALL`, `MEDIUM`, and `LARGE`; the paper uses `LARGE` with FP64.
The CPU driver writes `RESULTS.csv`, `manifest.json`, and per-case logs.

An existing ATen case can likewise be selected explicitly:

```
/usr/bin/python3 reproduce.py aten --stage plan --cases aten_mm --output results/runs/reuse-aten-mm
```

Replace `plan` with `all` to execute. This driver uses the dimensions and dtype
in [`aten/paper_shapes.json`](../aten/paper_shapes.json), performs the paper-input
correctness comparisons, and records five timing pairs. Case selection depends
on the suite; consult its driver help rather than assuming every suite accepts
the same options. Planning checks argument routing, not successful execution.

For custom ATen dimensions, copy the shape manifest and edit that case's
`dims`, related aliases, and descriptive metadata consistently, preserving its
dtype. Both lower-level drivers accept `--shape-specs PATH`:
[`aten_paper115_campaign.py`](../scripts/correctness/aten_paper115_campaign.py)
for raising/matching and
[`aten_fixture_batch.py`](../scripts/correctness/aten_fixture_batch.py)
for GPU build/run. Pass the same file to both and follow the existing commands
in [`aten/reproduce.py`](../aten/reproduce.py), including its environment setup
through [`common.py`](../scripts/artifact/common.py). The root ATen wrapper
currently fixes the paper manifest; it does not expose a custom-shape option.
Dimensions are applied to fresh source copies. Label custom datasets separately
from paper reproduction.

## Add a benchmark

1. Add the C input under the appropriate benchmark directory, along with its
   origin, license, dimensions, dtype, and numerical assumptions. Preserve any
   original extraction and record frontend adaptations separately.
2. Add a harness that initializes valid inputs and compares every output with
   the original program. For performance against a library, invoke its actual
   implementation as the baseline and preserve the same timing boundary.
3. Register the case in that suite's driver. For ATen, use the campaign's
   `--checklist` option with a CSV following the existing checklist schema;
   also provide argument/shape metadata, initialization, and a native ATen
   invocation in the fixture driver. Adding a C file alone is insufficient.
4. Run the shared compiler pipeline and retain the raised IR, matching proof,
   emitted API calls, correctness results, and timings. Recognition alone does
   not establish an executable replacement or a speedup.

Existing operations and backends can use the shared raising and matching
implementation. Unsupported IR may require general compiler support. Do not
add a recognizer keyed to the benchmark's name or source pattern.

## Import a library reference and bind its API

The compiler dependency owns specifications and runtime bindings. For a new
library operation:

1. Add a reference under `generic_solver/lib_v3/ref_c/` and its JSON manifest
   under `generic_solver/lib_v3/manifest/`. Record public upstream source hashes,
   any transliteration, argument order, types, layout, and numerical assumptions.
   The existing [Thrust fill reference](../third_party/Polygeist/generic_solver/lib_v3/ref_c/thrust_fill.c)
   and [manifest](../third_party/Polygeist/generic_solver/lib_v3/manifest/thrust_fill.json)
   illustrate the reference/API relationship.
2. Describe the target API and its argument binding in the manifest. If the
   runtime lacks that binding, implement an adapter that calls the actual
   library, with its required headers and linkage. A specification alone does
   not supply an executable backend.
3. Run `/usr/bin/python3 setup.py specs` with the compiler and source dependencies
   available. It uses the `lib_v3` catalog and shared raising recipe to generate
   MLIR and import metadata. Inspect stage validation results; do not hand-edit
   generated specifications or bypass validation for an unsupported stage.
4. Exercise a caller through raising, equality saturation, API emission, and
   full-output execution checks. Include shapes, layouts, and numerical edge
   cases relevant to the declared assumptions. Measure only after correctness
   passes. Test that invalid layouts or unsupported assumptions are rejected.

Manual work consists of preparing the reference and provenance, specifying the
API binding, supplying a runtime adapter when needed, and writing validation
cases. Matching uses the imported IR rather than a new operation-specific
matcher. When publishing an extension, record both repository revisions and
update the Polygeist submodule pin to the compiler commit containing it.

## Change hardware and retain useful results

Start with [`config.example.json`](../config.example.json): configure the SSH
alias, library locations, toolchain, and CPU affinity. Another GPU architecture
also requires appropriate compilation targets and compatible runtime libraries;
changing the SSH hostname alone does not establish portability.

Record hardware, clocks, load, software versions, source/compiler hashes, input
dimensions, numerical mode, and raw samples. Keep correctness failures visible
and report altered shapes or hardware as a new experiment. Use the
[timing estimates](../README.md#expected-experiment-time) to plan runs, but
measure their elapsed time on the new configuration. Separate unchanged paper
reproductions from extensions so their results remain interpretable.
