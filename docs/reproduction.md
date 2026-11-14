# Setup and execution

Run commands from the repository root. See the [README](../README.md) for
hardware specifications, experiment durations, and links to each experiment.

## Dependencies

Build the pinned Polygeist and LLVM/MLIR sources; system MLIR releases are not
interchangeable with this version. Install:

- Python 3.10 or later, `egglog==11.4.0`, and NumPy;
- CMake, Ninja, Git, GCC/G++, Clang/LLVM 14, and OpenBLAS development files;
- `gcc-aarch64-linux-gnu` and `g++-aarch64-linux-gnu` for Orin executables;
- CUDA 12.6, including nvcc, CCCL headers, and the AArch64 SBSA target;
- AArch64 cuBLAS, cuSOLVER, cuDNN, and cuTENSOR libraries for GPU experiments;
- an AArch64 OpenBLAS sysroot for PolyBench CPU;
- Docker for SpEQ and graph-size comparisons.

ATen additionally needs AArch64 LibTorch. Its plots use `pdflatex` with
PGFPlots and `standalone`, plus `pdftoppm`. PVA needs the separately installed [PVA SDK](../pva/README.md).
The [tested versions](../README.md#testing-systems) are listed in the README.

```
cp config.example.json config.json
export POLYGEIST_ROOT="$PWD/third_party/Polygeist"
export POLYGEIST_DEPENDENCY_ROOT="$PWD/third_party"
source "$POLYGEIST_ROOT/envsetup.sh"
python3 setup.py sources
python3 setup.py compiler
python3 setup.py specs
```

`config.json` is ignored by Git. Set its compiler, library, and sysroot paths
for your installation. The `environment` mapping supplies compiler variables
such as `CUDA`, `POLYGEIST_CUTENSOR_ROOT`, and `POLYGEIST_GGML_BUILD`.
Configuration keys can also be overridden with uppercase `LL_` environment
variables, for example `LL_SSH_HOST=orin`.

`sources` fetches the public revisions in `dependency_sources/repos.json` and
checks their hashes against the library reference manifests. `specs` raises
and validates the `lib_v3` references using the built compiler. Run it again
if you change the compiler or references. Library specifications and API
bindings live in `third_party/Polygeist/generic_solver/lib_v3/`.

## Execution on Orin

Figure 3 uses the Orin CPU; GPU experiments use its GPU. The host compiler
cross-compiles executables and the shared runner transfers them over SSH.
Set `ssh_host` to an alias in your SSH configuration, or `local` when running
on the target itself. Set `remote_library_path` to the target's CUDA, cuDNN,
cuTENSOR, and other shared-library directories.

Set the device's power mode and clocks before timing. The preflight records
clocks, load, temperature, and fan state and checks that the device is ready.
It does not change system settings. Run timing experiments sequentially on an
otherwise idle device. The paper used a fixed fan speed; automatic fan control
is a difference to record when comparing measurements.

## Comparison tools

Install only the tools needed for the selected experiments:

```
python3 setup.py kernelfarer
python3 setup.py ginsbach
python3 setup.py speq
python3 setup.py ggml
```

KernelFaRer and Ginsbach/IDL run on the x86 host. Their setup commands apply
the patches included beside their experiment drivers. Llama uses ggml's CUDA
implementation on Orin as its expert baseline.

SpEQ setup downloads its [published artifact](https://zenodo.org/records/10802555)
(6.3 GB) and builds the included LLVM fork in Docker. Allow additional disk
space and build time. The image preserves SpEQ's recognizer, library, LLVM
passes, and 5/1/3 saturation schedule; unrelated comparison tools are omitted.

## Commands and results

Use `python3 reproduce.py SUITE --output DIRECTORY` to build and run a suite.
Each experiment's README gives its command and explains the output files.
Use a new directory for each build. Runs record commands, source and compiler
hashes, intermediate IR, correctness checks, and raw timing samples.

Common stages are:

- `--stage plan`: print the command without running it;
- `--stage build`: compile the selected inputs;
- `--stage run`: execute the binaries in an existing output directory;
- `--stage summarize`: regenerate reports, where supported.

For EqSat and graph sizes, analysis takes place in `build`; `run` checks and
reports the analysis output. The CPU driver produces its report during `run`.
ATen has additional `match` and `plot` stages described in its README.

For a small host-only check:

```
python3 reproduce.py polybench_cpu --cases gemm --target host --size MINI --output results/runs/smoke
```

This checks compilation and execution. Figure 3 performance requires the
LARGE inputs on Orin. Report subsets, changed hardware, and changed dimensions
alongside the measurements. See [extending the experiments](extending-experiments.md)
for examples of changing inputs or adding a library reference.
