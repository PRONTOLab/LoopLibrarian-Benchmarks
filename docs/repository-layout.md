# Repository layout

Benchmark inputs are ordinary files owned by this repository. They do not
resolve through symbolic links into retained experiment results.

- `aten/inputs/`: all 598 extracted C kernels.
- `aten/harnesses/`: benchmark harnesses and the native CUDA comparison source.
- `aten/paper_cases.csv`: the 115-case paper cohort, with direct source paths.
- `aten/paper_shapes.json`: recorded dimensions and input-recipe metadata.
- `aten/metadata/`: extraction provenance and library/dispatch inventories.
- `mfem/inputs/`: source fixtures, harnesses, and the supported-case manifest.
- `llama/inputs/`: the inference fixture, harnesses, and operation manifest.
- `polybench/sources/`: PolyBench sources and utilities.
- `inputs/extracted/`: additional extracted source collections used by checks.
- `results/archive/`: retained measurements, IR, logs, and historical reports.
- `results/runs/`: ignored output directories for new experiments.

Each suite's `reproduce.py` is its entry point; the top-level `reproduce.py`
selects a suite and records the command and environment. Shared benchmark
harness code lives in `scripts/correctness/`. Compiler code and specifications
live in the pinned `third_party/Polygeist` submodule. Drivers reference those
tools directly; the benchmark scripts directory does not contain symlinks to
compiler modules.

`setup.py sources` copies the public cuDNN reference files bundled with the
pinned compiler into the dependency installation directory. These generated
copies are ignored by Git and checked against specification source hashes.
Existing copies with different contents are rejected rather than overwritten.

Saved reports retain historical source paths as provenance. `source_path()`
translates old ATen and MFEM fixture paths to their current source directories.
Current source lookup and the ATen paper-cohort manifest do not require the
archive. Historical analysis tools can still read archived measurements.

The layout tests check the complete ATen paper cohort and all supported MFEM
sources in a temporary package without `results/archive/`, and reject symlinked
benchmark source directories. Relocating files does not establish new matching,
correctness, or performance results; those require the benchmark drivers.
