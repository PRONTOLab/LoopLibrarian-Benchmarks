# Artifact for "LoopLibrarian: Automatically Targeting Library Functions as a Compiler ISA" (CGO '27)

This repository contains the artifact for the paper:
> **"LoopLibrarian: Automatically Targeting Library Functions as a Compiler ISA"**  
> *IEEE/ACM International Symposium on Code Generation and Optimization (CGO '27)*

Artifact DOI: [![DOI](https://zenodo.org/badge/1455872115.svg)](https://zenodo.org/badge/latestdoi/1455872115)

LoopLibrarian raises C computations to MLIR and matches them against
library specifications to generate library calls. The compiler is pinned under
`third_party/Polygeist`.

## Initial setup
We support three ways of setting up the experiment environment: a [pre-built Docker image](#pre-built-docker-image),
[building your own Docker image](#building-the-docker-image), and [manual setup](#manual-setup).
We recommend the pre-built image, as building the compiler requires building LLVM,
which takes a significant amount of time.

All three start from a checkout of this repository:

```
git clone https://github.com/PRONTOLab/LoopLibrarian-Benchmarks.git
cd LoopLibrarian-Benchmarks
```

### Pre-built Docker image

Requirements: x86-64 Linux, Docker, at least 32 GB RAM, and about 100 GB free disk space.

The image includes LLVM, MLIR, Polygeist, OpenBLAS, Python dependencies,
AArch64 cross-compilers, imported library specifications, plotting tools, the
native x86 CUDA libraries, and the KernelFaRer, IDL, and SpEQ comparison
tools. All experiments use this one image built by GitHub Actions.
Pull the image built from `main` and give it the local name used by the runner:

```
docker pull ghcr.io/prontolab/looplibrarian-benchmarks:latest
docker tag ghcr.io/prontolab/looplibrarian-benchmarks:latest looplibrarian:artifact
```

To use other sources, [build the image locally](#building-the-docker-image).

Then run the smoke test, which takes approximately 1–5 minutes and compares
every output of native and raised GEMM:

```
bash docker/run.sh polybench_cpu --cases gemm --target host --size MINI --output results/runs/smoke
```

`docker/run.sh` accepts the same suites and options as `reproduce.py`. In the
experiment commands below, replace `python3 reproduce.py` with `bash docker/run.sh`
to use Docker. Alternatively, use `bash docker/run.sh shell` for an interactive shell. 
To select another image, set `LL_DOCKER_IMAGE`. 
Docker runs as root; if necessary, restore result ownership:

```
sudo chown -R "$(id -u):$(id -g)" results/runs
```

GPU experiments on an x86 NVIDIA GPU server use the same prebuilt image pulled
above. Run the GPU smoke test directly:

```
bash docker/run.sh polybench_gpu --cases gemm --output results/runs/gpu-smoke
```

The host needs an NVIDIA driver compatible with CUDA 12.6 and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image already contains CUDA, cuDNN, cuTENSOR, PyTorch and the ggml baseline.
Library paths are configured inside the image; no host SDK mounts or manual
`library_path` setting are needed. The runner enables GPU access for execution;
`--stage build` does not require a GPU. Use `LL_DOCKER_GPUS=device=0` to select
one GPU on a shared server.

GPU experiments default to `--gpu-target x86_64` and `sm_80`. The chosen target
is saved with the experiment and reused by `--stage run`. Correctness checks and
five timing samples are the same on both platforms. Experiment commands compile
the benchmark programs using the compiler already in the image.

Use `--gpu-target jetson --stage build` for the separate
[Orin execution path](#execution-on-orin). A second machine is unnecessary for
native x86 GPU experiments.

### Build locally and execute on a remote x86 GPU server

The remote helper also defaults to `x86_64`. Build without a GPU, package the
executables, then use your SSH login to run them and retrieve the results:

```
bash docker/run.sh polybench_gpu --stage build --output results/runs/x86-gpu
python3 target.py pack --build polybench_gpu=results/runs/x86-gpu --output results/bundles/x86-gpu
python3 target.py remote --bundle results/bundles/x86-gpu --host user@gpu-server --results results/runs/remote-x86-gpu
```

The execution server needs Python 3.10 or later and the suite's runtime
dependencies (CUDA libraries, plus PyTorch for ATen). It does not need the
compiler. Use `--library-path` for libraries outside the system search paths.
When running a copied bundle inside an existing artifact container, set
`LL_ARTIFACT_CONFIG=/opt/looplibrarian/config.json` and invoke
`python3 /path/to/bundle/target.py run --bundle /path/to/bundle`; the helper uses
that container's configured runtime libraries. Bundles retain the build's target
and CUDA architecture unchanged. Execution restores those settings and rejects a
different target or an incompatible CPU architecture. Optional `--gpu-arch`
must match the saved architecture; choose it at build time for the execution
GPU. Jetson bundles require explicit `--gpu-target jetson`. Failed runs still
retrieve available diagnostics into `--results`; `transport.json` records both
the execution and retrieval status.

### Building the Docker image

This optional section is for contributors changing the compiler or image
configuration. Reviewers can use the prebuilt image above for CPU and native
x86 GPU experiments.

`Dockerfile` builds one image holding the compiler, the native x86 CUDA libraries
with the ggml and PyTorch comparison implementations, and the KernelFaRer,
IDL, and SpEQ comparison tools. Build locally if you change the compiler or need
another image revision. Builds use committed sources and exclude local
configuration and results:

```
bash docker/build.sh
```

For a custom GPU target, set the architecture when building the image and use
the same target for the experiments. For example, build the image and ggml
baseline for `sm_90` with:

```
LL_GPU_ARCH=90 bash docker/build.sh
```

Then pass `--gpu-arch sm_90` to the experiment commands.

Allow 1–4 hours for a first build, plus 20–60 minutes and additional disk space
for CUDA and PyTorch. LLVM and Polygeist took about 48 minutes with `BUILD_JOBS=8`
on the listed host; lower `BUILD_JOBS` on smaller machines. The image imports
specifications and runs script tests. IDL and KernelFaRer build with
`COMPARISON_JOBS=4` by default. Later builds reuse unchanged compiler layers.
SpEQ is built in the same Docker build and adds a 6.3 GB source download and
roughly 1–2 hours on a fresh build.

### Manual setup

These commands target Ubuntu 22.04 on x86-64, matching the image's base system.
Run them from the repository root. Install the compiler and comparison-tool
build dependencies:

```
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ca-certificates ccache clang-14 cmake curl file gfortran git \
  graphviz libffi-dev libncurses-dev libopenblas-dev libxml2-dev lld-14 \
  llvm-14-dev ninja-build openssh-client pkg-config python3 python3-dev \
  python3-pip python3-venv time unzip xxd zlib1g-dev zstd
```

Install the ATen plotting tools (`pdflatex`, PGFPlots, `standalone`, and
`pdftoppm`) if you will generate figures:

```
sudo apt-get install -y --no-install-recommends \
  texlive-latex-extra texlive-pictures poppler-utils
```

Create a Python 3.10 environment for LoopLibrarian:

```
python3 -m venv build/python
source build/python/bin/activate
python3 -m pip install egglog==11.4.0 numpy==1.26.4
```

Keep this environment active for the following commands; reactivate it with
`source build/python/bin/activate` in a new shell. SpEQ's setup creates its own
environment with its required older egglog version, so do not install
`dependency_sources/requirements.txt` into this environment.

Initialize the compiler checkout, download the pinned sources, and build the
compiler and library specifications:

```
git submodule update --init third_party/Polygeist
cp config.example.json config.json
export POLYGEIST_ROOT="$PWD/third_party/Polygeist"
export POLYGEIST_DEPENDENCY_ROOT="$PWD/third_party"
# Edit config.json for the installed libraries and SDKs.
python3 dependency_sources/prepare_cuda_headers.py
python3 setup.py sources
python3 setup.py compiler
python3 setup.py specs
python3 reproduce.py polybench_cpu --cases gemm --target host --size MINI --output results/runs/smoke
```

`config.json` is ignored by Git. Its `environment` mapping supplies compiler
variables such as `CUDA` and `POLYGEIST_CUTENSOR_ROOT`. Configuration keys can
also be overridden with uppercase `LL_` variables, for example `LL_JOBS=4`.

`sources` fetches pinned public dependencies and verifies the reference hashes.
The header preparation command checks the public package hashes; execution still
requires the target SDKs. `compiler` builds the public LLVM revision in
[llvm.json](dependency_sources/llvm.json), applies the included `cf.switch` parser
patch, and verifies the source-tree hash before building Polygeist. System MLIR
releases are not interchangeable with this compiler. `specs` raises and validates
the `lib_v3` references; rerun it after changing the compiler or references.

Allow 15–60 minutes for source downloads, roughly 4–12 hours for a clean native
compiler build, and 2–10 minutes for specification import.

#### Native CUDA on x86

For native GPU execution, the host also needs a working NVIDIA driver compatible
with CUDA 12.6. With the Python environment above active, install the same CUDA
toolkit components and Python GPU packages as the image. The repository setup
follows the [CUDA 12.6 installation guide](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html):

```
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/looplibrarian-cuda-keyring.deb
sudo dpkg -i /tmp/looplibrarian-cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  cuda-nvcc-12-6=12.6.85-1 cuda-libraries-dev-12-6=12.6.3-1
python3 -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install cutensor-cu12==2.3.1
export PATH="/usr/local/cuda/bin:$PATH"
```

The [PyTorch CUDA 12.6 package](https://pytorch.org/get-started/previous-versions/#v290)
supplies the native ATen baseline and cuDNN dependencies. Configure their paths
in the fresh `config.json` created above and build the ggml baseline:

```
python3 docker/configure_gpu.py --gpu-arch 86
python3 setup.py ggml
python3 reproduce.py polybench_gpu --cases gemm --output results/runs/native-gpu-smoke
```

Architecture `86` is for RTX 3090; select the architecture of your GPU. The
configuration helper also works outside Docker: it locates the active Python
environment's PyTorch, cuDNN and cuTENSOR installations and writes their library
paths to `config.json`. It replaces that file's `environment` mapping, so add
any optional SDK settings afterward.

#### Cross-compiling for Orin

On the x86 build host, additionally install the AArch64 C/C++ compilers:

```
sudo apt-get install -y --no-install-recommends \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

The cross build also needs an AArch64 OpenBLAS sysroot, CUDA 12.6 with its SBSA
target libraries, and target-compatible cuDNN, cuTENSOR and, for ATen, LibTorch.
The x86 Python GPU packages above do not supply those ARM libraries. Install
the target SDKs using their platform-specific distributions, then configure the
paths described under [Execution on Orin](#execution-on-orin). The pinned public
CCCL and reference headers are prepared by the `sources` and header commands
above. MFEM execution additionally needs cuTensorNet; PVA needs the separately
installed [PVA SDK](https://developer.nvidia.com/embedded/pva) and its operator
libraries.

### Comparison tools

The Docker image already contains KernelFaRer, IDL, and SpEQ, including
SpEQ's reference sources, LLVM fork, and isolated Python environment. Run the
comparison experiments directly with `bash docker/run.sh`; no additional setup,
image build, or Docker socket mount is needed.

For a native installation, build only the tools needed for your experiment:

```
python3 setup.py kernelfarer
python3 setup.py ginsbach
python3 setup.py speq
```

These use the original projects and the included compatibility patches.
IDL builds its own LLVM fork; allow about 30 minutes with four build jobs.
The native SpEQ setup downloads the [published artifact](https://zenodo.org/records/10802555)
(6.3 GB), verifies and extracts the needed sources into `third_party/speq`, and
builds its LLVM tools and Python environment there. Use Python 3.10 with its
`venv` module and the build packages listed in the root Dockerfile; allow about
1–2 hours.
SpEQ retains its recognizer, library, LLVM passes, and 5/1/3 saturation schedule.
GitHub Actions checks linearized GEMM recognition and the 370-node SpEQ graph
inside the combined image before publishing it.


## Experiments and estimated runtimes

Times include benchmark compilation and execution on the systems below, but exclude
setup. They depend on machine load. Run GPU timing experiments one at a time.

| Experiment | Paper result | Approximate time after setup |
|---|---|---|
| [PolyBench CPU](#polybench-cpu-figure-3) | Figure 3 | 40 minutes–2 hours |
| [PolyBench GPU](#polybench-gpu-figure-4) | Figure 4 | 20–50 minutes |
| [Llama](#llama-section-422) | Section 4.2.2 | 15–40 minutes |
| [ATen matching](#aten-matching-table-5) | Table 5 | 1–3 hours |
| [ATen performance](#aten-performance-figure-5) | Figure 5 | 1–4 hours |
| [MFEM](#mfem-table-6) | Table 6 | 15–45 minutes |
| [Equality saturation](#equality-saturation-table-10) | Table 10 | 30 minutes–2 hours |
| [Graph sizes](#gemm-e-graph-sizes-section-443) | Section 4.4.3 | 2–10 minutes |
| [IDL](#idl-section-441) | Section 4.4.1 | 10–30 minutes |
| [KernelFaRer](#kernelfarer-table-9) | Table 9 | 30–90 minutes |
| [PVA](#pva-table-7) | Table 7 | 15–30 minutes |
| [Compilation cost](#compilation-cost-table-8) | Table 8 | 6–24 hours, estimated |

## Testing systems
The paper's measurements used the following build host and Orin target.
Native x86 GPU experiments build and run on one machine; these two machines are
only needed to repeat the original cross-compilation setup.

| Component | Build host | Execution target |
|---|---|---|
| Machine | KVM VM, 24 Intel Xeon Cascade Lake vCPUs | Jetson AGX Orin Developer Kit, 12 CPU cores |
| RAM | 62 GiB | 61.4 GiB |
| GPU | Not used | Integrated Orin GPU, compute capability 8.7 |
| OS | Ubuntu 22.04.1, Linux 5.15.0-151-generic | Ubuntu 24.04.4, Linux 6.8.12-1021-tegra |
| Clocks and power | Shared VM | CPU 2.2016 GHz; GPU 1.3005 GHz; MAXN mode 0 |

Figure 3 uses CPU 0 on Orin with one OpenBLAS thread. PolyBench GPU, Llama,
ATen, and MFEM execute on Orin. KernelFaRer, IDL, equality saturation, graph
measurements, and compilation measurements use the x86 host. PVA requires the
Orin PVA accelerator and a separately installed SDK.

The native x86 GPU path was also tested on an Intel Core i7-8700K host
(6 cores, 12 threads) with an NVIDIA GeForce RTX 3090 (24 GiB, compute capability
8.6). The prebuilt combined Docker image passed PolyBench GPU GEMM and ATen
conv3d correctness and timing checks on this GPU with NVIDIA driver 580.178.04.

## Execution on Orin

### Local execution
This optional path runs the compiler on x86, packages the resulting ARM
executables, and runs them on a Jetson. It requires target SDKs
on the build host and compatible runtime libraries on the Jetson. Set
`gpu_target` to `jetson` in its configuration and set `library_path` to the
library directories on that device. The native x86 GPU Docker path above does
not need these settings.

The x86 image can cross-compile with `--stage build`. Supply `config.docker.json`
with paths as seen inside the container, such as `/sdk/cuda/bin/nvcc`,
`/sdk/cuda/targets/sbsa-linux`, `/sdk/cutensor`, and `/sdk/openblas`:

```
cp config.example.json config.docker.json
# Edit config.docker.json to use the paths inside /sdk.
export LL_DOCKER_CONFIG="$PWD/config.docker.json"
export LL_DOCKER_SDK_ROOT=/path/to/installed-sdks
bash docker/run.sh polybench_gpu --gpu-target jetson --stage build --output results/runs/fig4
```

SDKs are mounted read-only and excluded from the image. Package the built
experiment on the build host:

```
python3 target.py pack --build polybench_gpu=results/runs/fig4 --output results/bundles/gpu
```

Copy this bundle to any directory on Orin, then run it there:

```
python3 target.py run --bundle . --gpu-target jetson --library-path /usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
```

The bundle contains executables, input sources, validation scripts, and unchanged
build records. It needs Python 3.10 or later and the target runtime libraries. 
`--verify-only` checks the bundle's hashes on either machine. Outputs and logs 
are written to `runs/` inside the bundle.

Alternatively, transfer and execute on a machine you control through an existing
SSH configuration. This command also retrieves the results if an experiment fails:

```
python3 target.py remote --bundle results/bundles/gpu --gpu-target jetson --host user@orin --results results/runs/orin-gpu
```

Use `--jump-host user@gateway` if needed and `--library-path` for additional
target libraries. On Orin, select the runtime libraries supplied for its JetPack
installation, including cuBLAS; the cross-compilation SDK is for building only.
Authentication uses your SSH keys or agent. If you use a local `SSH_ASKPASS` helper,
pass `--ssh-option BatchMode=no`; other OpenSSH settings can be supplied with
repeated `--ssh-option KEY=VALUE` arguments. No access to the authors' machines is
needed or provided.

### Building and packaging multiple suites

The following example bundles PolyBench GPU, Llama, MFEM, PVA, and the ATen
Figure 5 selection. Use the Docker configuration and SDK mount above, with these
additional target dependencies configured before building:

- **Llama:** set `environment.POLYGEIST_GGML_BUILD` and run `python3 setup.py ggml`
  to build the pinned CUDA baseline. In Docker, use
  `/opt/looplibrarian/build/ggml-aarch64` and `LL_GPU_TARGET=jetson bash docker/run.sh setup ggml`.
- **ATen:** set `environment.ATEN_NATIVE_TORCH_ROOT`, `ATEN_NATIVE_CUDA_ROOT`, and
  `POLYGEIST_CUTENSOR_ROOT` to the target-compatible installations. LibTorch is
  supplied separately. Jiterator tests also need CUDA 12.6 NVRTC and matching
  `libnvrtc-builtins` in the target's `library_path`.
- **MFEM:** set `cutensornet_root` to the target-compatible cuTensorNet SDK as
  seen inside the container, for example `/sdk/cuquantum`.
- **PVA:** install the SDK and operator libraries separately. Set
  `environment.PVA_OPERATOR_INCLUDE`, `NVCV_INCLUDE`, `CUPVA_SDK_ROOT`, and
  `PVA_LIB_STAGE` to their installed paths. SDK sources and headers are not
  redistributed here.

On the x86 build host, run these commands from the repository root, using fresh
output directories. The first command prepares Llama's ggml baseline:

```
LL_GPU_TARGET=jetson bash docker/run.sh setup ggml
bash docker/run.sh polybench_gpu --gpu-target jetson --stage build --output results/runs/orin-polybench-gpu
bash docker/run.sh llama --gpu-target jetson --stage build --output results/runs/orin-llama
bash docker/run.sh mfem --gpu-target jetson --stage build --output results/runs/orin-mfem
bash docker/run.sh pva --gpu-target jetson --stage build --output results/runs/orin-pva
bash docker/run.sh aten_figure5 --gpu-target jetson --stage build --output results/runs/orin-aten-paper
```

Package the successful builds inside the container so the command can access
the four ggml shared libraries in its persistent build volume. The bundle is
written into the host checkout:

```
LL_GPU_TARGET=jetson bash docker/run.sh shell -c 'python3 target.py pack \
  --build polybench_gpu=results/runs/orin-polybench-gpu \
  --build llama=results/runs/orin-llama \
  --build mfem=results/runs/orin-mfem \
  --build pva=results/runs/orin-pva \
  --build aten=results/runs/orin-aten-paper \
  --ggml-build /opt/looplibrarian/build/ggml-aarch64 \
  --output results/bundles/orin-all'
python3 target.py run --bundle results/bundles/orin-all --verify-only
```

Vendor SDKs and their runtime libraries must be installed separately on the
Jetson. Copy the bundle to your device, replacing `user@orin` with your login:

```
scp -r results/bundles/orin-all user@orin:~/looplibrarian-orin-all
```

On the Jetson, run the following commands. Replace the `/opt/` library
directories with the installed target paths for cuTensorNet, cuTENSOR,
LibTorch, and PVA:

```
cd ~/looplibrarian-orin-all
python3 target.py run --bundle . --gpu-target jetson \
  --library-path /usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/opt/cuquantum/lib:/opt/cutensor/lib:/opt/libtorch/lib:/opt/pva/lib
```

All five suites run serially and write results under `runs/` in the bundle.
Each retains its correctness checks and timing protocol, including five
measured samples where the experiment measures performance.

## Running experiments

Use a fresh output directory for each build. The drivers save generated IR,
commands, source and executable hashes, output comparisons, and raw timing samples.
A completed command does not mean all cases passed: inspect its summary for failed
checks, missing matches, and differences from the published results.

Common stages are `--stage plan` to print a command, `build` to compile, and
`run --output PATH` to execute an existing build. `summarize` regenerates reports
where supported. Equality saturation and graph-size analysis happen during
`build`; their `run` stages report the analysis. PolyBench CPU writes its table
during `run`. ATen also provides `match`, `report`, and `plot` stages.

### PolyBench CPU (Figure 3)

```
python3 reproduce.py polybench_cpu --output results/runs/fig3
```

Run the 17 FP64 LARGE programs on Orin CPU 0 with one OpenBLAS thread, comparing
original C with complete LoopLibrarian programs. Each uses five warmups and five
measurements; report the minimum. Every output must pass `atol=rtol=1e-8`.
`results.json` contains checks, samples, and paper speedups; `RESULTS.csv` contains
the timing table. Select a case with `--cases gemm`. 

### PolyBench GPU (Figure 4)

```
python3 reproduce.py polybench_gpu --output results/runs/fig4
```

Run 12 FP64 LARGE kernels on the selected NVIDIA GPU. Five warmups precede five measurements; report
the minimum CUDA-event compute time, excluding allocation and initial/final
transfers. Synchronized end-to-end time is recorded separately. Every output
must pass `atol=rtol=1e-8` before timing is accepted.

`results.csv` contains timings and speedups; per-case files retain checks and a
separate vendor-call trace. These are separate-kernel harnesses: ten native bodies
come from pinned PolyBenchGPU, while correlation and covariance use the paper's
CUDA adaptations. Select a subset with `--cases gemm`.

### Llama (Section 4.2.2)

The Docker image includes the native x86 ggml baseline. For Jetson, first build its ggml baseline
as described under [Execution on Orin](#execution-on-orin).

```
python3 reproduce.py llama --output results/runs/llama
```

Check 19 operation fixtures and FP32, one-token, one-layer inference against ggml
CUDA. `coverage.json` separates recognition from executable lowering; the paper
reports 14/19. `summary.json` records five timing samples and compares all 32,000
logits with `atol=0.001, rtol=0.0001`. The reported LoopLibrarian runtime is
12.826 ms. Each run includes the native CPU program, LoopLibrarian, and the
hand-optimized ggml CUDA baseline. The summary reports their minimum and median
times. LoopLibrarian records both the GPU interval used for the paper comparison
and synchronized wall time; ggml reports synchronized wall time.
The driver measures resident GPU execution with CUDA graph replay
enabled by default and records that setting with the timings. Set
`LL_CUDA_GRAPH=0` to disable replay for a diagnostic run. The separate
serial CPU comparison may fail its strict tolerance.
Fixtures in `llama/inputs` include frontend adaptations such as split RoPE and
branchless masking.

### ATen matching (Table 5)

[table5.py](aten/table5.py) runs matching across all 598 extracted/transliterated
ATen C inputs listed in [all_cases.csv](aten/all_cases.csv), using the tensor
sizes in those source files. It reports complete matches across the full set
and the subset of 52 inputs whose native implementations call external libraries:

```
bash docker/run.sh aten_table5 --output results/runs/aten-table5
```

For a native installation, use
`python3 aten/table5.py --output results/runs/aten-table5` instead.
This experiment needs no GPU and does not execute or time kernels. A complete
match requires a proven library replacement, successful API lowering, and no
remaining computation loops. Matching takes an estimated 1–3 hours; `--jobs`
controls matching parallelism.

`table5.json` contains the aggregate counts, including incomplete and unevaluated
cases; `table5.csv` records every input. `matching/` retains generated IR and
proofs, and `summary.json`/`summary.csv` retain per-case status. To regenerate
the coverage reports from saved matching results:

```
bash docker/run.sh aten_table5 --stage report --output results/runs/aten-table5
```

### ATen performance (Figure 5)

Of the 351 matched ATen kernels, we evaluate GPU performance for 113 selected
non-pointwise kernels in Figure 5. [figure5.py](aten/figure5.py) uses the inputs
in [paper_cases.csv](aten/paper_cases.csv) and the performance-benchmark tensor
sizes in [paper_shapes.json](aten/paper_shapes.json). These sizes can differ
from the source sizes used for Table 5:

```
bash docker/run.sh aten_figure5 --output results/runs/aten-figure5
```

For a native installation, use
`python3 aten/figure5.py --output results/runs/aten-figure5` instead.
This matches, builds, checks outputs, collects five alternating synchronized
LoopLibrarian/native ATen CUDA timing pairs after five warmups, and plots the
validated results. Timings use the minimum of five samples per implementation.
Allow an estimated 1–4 hours; use `--cases aten_mm` for one case (about 2–10
minutes). `--jobs` controls compilation parallelism; GPU timings run serially.

Outputs include `matching/` (IR and proofs), `gpu/` (builds, checks and raw timing
samples), `gpu/report.csv`, and `summary.json`. Figure 5 is written to
`plots/aten-current.pdf` and `.png`, with its data in
`plots/aten-current-data.csv`. The caption reports the number of validated
timing pairs.

To resume a completed matching stage, run an existing build, or regenerate the
graph without new GPU execution, use the applicable command:

```
bash docker/run.sh aten_figure5 --stage build --output results/runs/aten-figure5
bash docker/run.sh aten_figure5 --stage run --output results/runs/aten-figure5
bash docker/run.sh aten_figure5 --stage plot --output results/runs/aten-figure5
```

Resumed stages retain the saved cases, tensor sizes and correctness policy.
Use separate output directories for Table 5 and Figure 5. Matching options and
published measurements are in [matching_options.json](aten/matching_options.json)
and [paper_timings.csv](aten/paper_timings.csv). Add
`--paper-figure path/to/figure.pdf` for a side-by-side comparison.

### MFEM (Table 6)

For Table 6, run compiler matching without a GPU:

```
python3 reproduce.py mfem --stage match --output results/runs/mfem-matching
```

`matching-table.csv` groups emitted contraction regions by operator family;
`matching-summary.json` retains the selected libraries and PCG vector calls.
This checks all 20 operator extractions, including the two elasticity cases
reported as unmatched, and the PCG application. `mfem/published-matching.json`
contains the paper's per-input counts and their source revision. The current
compiler covers the 128 previously matched contraction bodies and emits 135
cuTensorNet contractions. It additionally recognizes two contractions in value
integration, two in mass application, and three in 3D div-div.

Optional execution checks require the cuTensorNet SDK. Set `cutensornet_root`
in `config.json` or `POLYGEIST_CUTENSORNET_ROOT` to its installation directory.

```
python3 reproduce.py mfem --output results/runs/mfem
```

Run the 18 supported contraction fixtures (`NE=1024`) and integrated PCG updates
(`NE=2`) on the configured GPU target. Every output is compared with the original computation using
`1e-11 + 1e-10 * abs(expected)`. `summary.json` aggregates results; each case has
`result.json` and `run.log`. This validates execution, without a timing comparison.

### Equality saturation (Table 10)

Using the prebuilt image (native installations need [SpEQ setup](#comparison-tools)):

```
bash docker/run.sh eqsat --output results/runs/eqsat
```

Compile all 30 PolyBench programs with equality saturation enabled and disabled,
then compare recognition with SpEQ on the x86 host. We were able to raise
19 programs with equality saturation and 11 without, out of the same 30.

`summary.json` reports recognition and executable lowering separately, including
per-program failures. 
SpEQ's reported single match uses GEMM with linearized indexing; results on
unmodified PolyBench are also recorded.

### GEMM e-graph sizes (Section 4.4.3)

Using the prebuilt image (native installations need [SpEQ setup](#comparison-tools)):

```
bash docker/run.sh graph_sizes --output results/runs/graphs
```

`summary.json` reports the counts, with serialized LoopLibrarian graphs and SpEQ
DOT files beside it. The reported result is 370 SpEQ nodes versus 21 LoopLibrarian
nodes.

### IDL (Section 4.4.1)

Using the prebuilt image (native installations need [IDL setup](#comparison-tools)):

```
bash docker/run.sh ginsbach --output results/runs/idl
```

Run IDL on all 30 PolyBench programs and execute programs containing replacements
on x86. Expect four detections: one in correlation, two in covariance, and one in
trmm. All three programs fail the full-output comparison; reproducing those
failures is the reported result. `summary.json` records detections and checks.
The driver uses the original allocation helpers in
`inputs/polybench_original_utilities`; changing them affects IDL recognition.

### KernelFaRer (Table 9)

Using the prebuilt image (native installations need [KernelFaRer setup](#comparison-tools)):

```
bash docker/run.sh kernelfarer --output results/runs/table9
```

Compare native compilation, KernelFaRer, and LoopLibrarian for complete FP32 LARGE
GEMM, SYR2K, 2mm, and 3mm programs on x86.
Each implementation has five warmups and five measurements; report the minimum.
`summary.csv` contains comparisons; `timing.json` retains samples.

In that program order, the paper reports LoopLibrarian speedups of 8.6/366/115/93
and KernelFaRer speedups of 8.6/346/113/112. 

### PVA (Table 7)

Table 7 runs 18 cases across six PVA operator families on Jetson Orin. Install
the PVA SDK, operator headers and libraries, and NVCV headers separately, then
build and package the cases below. Follow [Execution on Orin](#execution-on-orin)
to run the bundle. `summary.json` records execution checks and output comparisons;
per-case results retain numerical differences and comparison criteria. This
experiment reports correctness and execution coverage without timings.

```
export LL_DOCKER_PVA_OPERATOR_INCLUDE=/path/to/operator/include
export LL_DOCKER_NVCV_INCLUDE=/path/to/nvcv/include
export LL_DOCKER_CUPVA_SDK_ROOT=/path/to/cupva-sdk
export LL_DOCKER_PVA_LIB_STAGE=/path/to/aarch64-pva-libraries
bash docker/run.sh pva --gpu-target jetson --stage build --vendor-compatibility --output results/runs/pva
python3 target.py pack --build pva=results/runs/pva --output results/pva-bundle
```

### Compilation cost (Table 8)

```
python3 reproduce.py compilation_cost --output results/runs/compilation-cost
```

[inputs.json](compilation_cost/inputs.json) lists 199 inputs: 30 PolyBench,
23 Llama (19 operation fixtures, extended forward, and three llama2.c functions),
115 ATen, and 31 MFEM. By default, measure compilation from C through raising and
verified equality-saturation matching, sequentially three times per input with
all rewrites enabled (`--repetitions 3 --modes full`): 597 compilations. Each run
stops before ABI lowering, LLVM code generation, runtime-object compilation, and
linking. The estimated 6–24 hours extrapolates the 32–138 seconds through matching
recorded in a four-input pilot; duration depends on the inputs and machine load.
A small check is:

```
python3 reproduce.py compilation_cost --cases aten/aten_avg_pool2d --repetitions 1 --output results/runs/compilation-check
```

`compilation-table.csv` reports median compilation times by suite; `measurement-tables.csv`
contains stage times, proof measurements, and peak memory. `summary.json` retains
raw repetitions and failures.

The default uses three repetitions, as in the paper. Frontend translation,
raising, and matching are repeated; parsed reference catalogs are cached per
configuration. Peak RSS is the maximum for one process, not concurrent workers' sum.

To resume, use the suite driver directly with unchanged inputs and compiler:

```
python3 compilation_cost/reproduce.py --resume --output results/runs/compilation-cost
```

Use `python3 reproduce.py compilation_cost --stage summarize --output results/runs/compilation-cost`
to report available data without more compilations. Summaries retain the recorded
measurement boundary; older complete-build timings remain labeled separately.

## Extending the experiments

Use a new output directory and record changes to dimensions, dtypes, or hardware.
For PolyBench, select `--cases gemm --size SMALL`; supported sizes are `MINI`,
`SMALL`, `MEDIUM`, and `LARGE`. Paper Figure 3 uses FP64 LARGE inputs on Orin.

For ATen dimensions, copy [paper_shapes.json](aten/paper_shapes.json) and update
`dims` and related aliases consistently. Pass it through `--shape-specs` to both
[aten_paper115_campaign.py](scripts/correctness/aten_paper115_campaign.py) and
[aten_fixture_batch.py](scripts/correctness/aten_fixture_batch.py); the top-level
ATen command does not expose this option. [aten/reproduce.py](aten/reproduce.py)
shows the invocations.

To add a benchmark:

1. Add C source, origin, license, dimensions, dtype, and frontend adaptations to
   its experiment directory.
2. Add a harness checking every output against the original. For timings, call
   the real baseline implementation using the same timing boundary.
3. Register the case in the driver. ATen needs a checklist following
   [all_cases.csv](aten/all_cases.csv), shape/argument metadata, and a native
   ATen invocation; both lower-level drivers accept `--checklist`.
4. Check compilation, generated library calls, and output correctness before timing.

## Extending LoopLibrarian

LoopLibrarian's compiler lives in `third_party/Polygeist`. Extensions can improve
how programs are raised to MLIR, how equivalent computations are matched, or
which library implementations are available.

### Improving compiler passes

When a computation cannot be raised, inspect the intermediate MLIR to identify
the unsupported loop, memory access, or control flow. Start with
[RaiseToLinalg.cpp](third_party/Polygeist/lib/polygeist/Passes/RaiseToLinalg.cpp)
for converting loops to Linalg, and
[LinalgDebufferize.cpp](third_party/Polygeist/lib/polygeist/Passes/LinalgDebufferize.cpp)
for converting buffer operations to tensor operations. Add a small regression
case under `test/polygeist-opt/`. Application code and library references use
the shared [raising recipe](third_party/Polygeist/scripts/correctness/lib_v2/recipe.py);
rebuild the compiler and regenerate specifications after changing these passes.

### Improving matching

When raising succeeds but a library replacement is missing, inspect the generated
IR and match report. The matching entry point,
[spec_match.py](third_party/Polygeist/scripts/correctness/spec_match.py), invokes
the [Linalg matcher](third_party/Polygeist/scripts/correctness/lib_v2/match_v2.py)
and [generic region matcher](third_party/Polygeist/scripts/correctness/region_match/matcher.py).
Extend their handling of dataflow, indexing, or equality-saturation rules while
preserving the selected numerical assumptions. Add tests for both equivalent
computations that should match and similar computations that must be rejected.
Library replacements must be justified by imported reference semantics.

### Adding library references

1. Add a C reference under `generic_solver/lib_v3/ref_c/` and a manifest under
   `generic_solver/lib_v3/manifest/`. Record its public source, license,
   adaptations, argument types, layouts, numerical assumptions, and target API.
   The [Thrust fill reference](third_party/Polygeist/generic_solver/lib_v3/ref_c/thrust_fill.c)
   and [manifest](third_party/Polygeist/generic_solver/lib_v3/manifest/thrust_fill.json)
   provide an example.
2. Declare the API bindings and add a runtime adapter where needed.
3. From the artifact root, run `python3 setup.py specs` to raise and validate the
   reference through the shared pipeline.
4. Check generated calls and full outputs for supported shapes, layouts, and
   data types, including rejection of unsupported inputs.

See the [library specification documentation](third_party/Polygeist/generic_solver/lib_v3/README.md)
for import and validation details. When distributing compiler extensions, commit
them in Polygeist and update this artifact's submodule revision.

## Repository files and licenses

- Benchmark directories contain inputs, harnesses, manifests, and experiment drivers.
- `third_party/` contains the pinned compiler and downloaded dependencies;
  `dependency_sources/` records their revisions, hashes, and build configuration.
- `scripts/artifact/` provides configuration, logging, and local execution support;
  `scripts/correctness/` provides shared compilation and comparison code.
- `tests/` checks drivers and manifests; run `python3 -m unittest discover -s tests`.
- `results/runs/` receives generated results and is ignored by Git.
- `docs/system-specifications.json` records the tested systems in detail.

Compiler-derived scripts retain their source repository's [license](LICENSE).
Benchmark and dependency sources retain upstream licenses and notices; headers
and manifests identify their origins and adaptations. Public revisions are in
[repos.json](dependency_sources/repos.json); Polygeist reference manifests record
source hashes. Proprietary libraries and SDK components are installed separately
and are not redistributed here.
