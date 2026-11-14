# Setup and execution

Run commands from the repository root. See the [README](../README.md) for
hardware specifications, experiment durations, and links to each experiment.
The [Docker setup](../docker/README.md) automates the compiler and host dependencies.

## Dependencies

Initialize the compiler with `git submodule update --init third_party/Polygeist`.
Build the pinned Polygeist and LLVM/MLIR sources; system MLIR releases are not
interchangeable with this version. Install:

- Python 3.10 or later, `egglog==11.4.0`, and NumPy;
- CMake, Ninja, Git, GCC/G++, Clang/LLVM 14, OpenBLAS development files, `file`, and `zstd`;
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
python3 dependency_sources/prepare_cuda_headers.py
python3 setup.py sources
python3 setup.py compiler
python3 setup.py specs
```

`config.json` is ignored by Git. Set its compiler, library, and sysroot paths
for your installation. The `environment` mapping supplies compiler variables
such as `CUDA`, `POLYGEIST_CUTENSOR_ROOT`, and `POLYGEIST_GGML_BUILD`.
Configuration keys can also be overridden with uppercase `LL_` environment
variables, for example `LL_JOBS=4`.

`sources` fetches the public revisions in `dependency_sources/repos.json` and
checks their hashes against the library reference manifests. It also fetches
LAPACK sources used to check BLAS/LAPACK argument types. `specs` raises
and validates the `lib_v3` references using the built compiler. Run it again
if you change the compiler or references. Library specifications and API
bindings live in `third_party/Polygeist/generic_solver/lib_v3/`.

`prepare_cuda_headers.py` downloads the public header packages listed in
[`cuda-headers.json`](../dependency_sources/cuda-headers.json) and checks archive
hashes. These supply source imports and API checks; execution still uses the
CUDA 12.6 target SDK.

`setup.py compiler` fetches the public LLVM revision in
`dependency_sources/llvm.json` and applies `llvm-switch-operands.patch`, which
allows numbered SSA results in `cf.switch` operands. Setup verifies the complete
source-tree hash against the tested LLVM tree and refuses unrelated local changes.

## Execution on Orin

Figure 3 uses the Orin CPU; GPU experiments use its GPU. Execution is local:
run the experiments on your own compatible device. Set `library_path` to its
CUDA, cuDNN, cuTENSOR, and other shared-library directories. No access to the
authors' machines is required or provided.

The x86 container can cross-compile using `--stage build`. It cannot execute
the resulting AArch64 binaries. Run `--stage run` directly on an AArch64 target
with the checkout, dependencies, and build results available at the paths
recorded during compilation. The current result records retain absolute paths;
moving a build to a different directory is not supported. Alternatively, use
the native setup on the target and build and run there. A native AArch64
compiler build and an AArch64 Docker image have not yet been validated.

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
