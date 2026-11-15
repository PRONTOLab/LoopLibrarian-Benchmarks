# LoopLibrarian artifact

This repository contains the benchmarks and reproduction scripts for
**LoopLibrarian: Automatically Targeting Library Functions as a Compiler ISA**
(CGO ’27). LoopLibrarian raises C computations to MLIR and matches them against
library specifications to generate library calls. The compiler is pinned under
`third_party/Polygeist`.

All setup and experiment instructions are below. Run commands from the repository root.

- [Experiments and estimated runtimes](#experiments-and-estimated-runtimes)
- [Testing systems](#testing-systems)
- [Setup and smoke test](#setup-and-smoke-test)
- [Execution on Orin](#execution-on-orin)
- [Running experiments](#running-experiments)
- [Extending the experiments](#extending-the-experiments)
- [Repository files and licenses](#repository-files-and-licenses)
- [Building and publishing Docker images](#building-and-publishing-docker-images)

## Experiments and estimated runtimes

Times include benchmark compilation and execution on the systems below, but exclude
setup. They depend on machine load. Run GPU timing experiments one at a time.

| Experiment | Paper result or comparison | Approximate time after setup |
|---|---|---|
| [PolyBench CPU](#polybench-cpu-figure-3) | Figure 3: 17 programs, OpenBLAS on Orin CPU | 40 minutes–2 hours |
| [PolyBench GPU](#polybench-gpu-figure-4) | Figure 4: 12 kernels, CUDA on Orin | 20–50 minutes |
| [Llama](#llama) | 19-operation coverage and one-layer inference versus ggml | 15–40 minutes |
| [ATen](#aten-table-5-and-figure-5) | Table 5 coverage; Figure 5's 115 timing cases | 1–3 hours for the 115 cases |
| [MFEM](#mfem) | Contraction fixtures and PCG updates | 15–45 minutes |
| [Equality saturation](#equality-saturation) | Matching with/without equality saturation, versus SpEQ | 30 minutes–2 hours |
| [Graph sizes](#gemm-e-graph-sizes) | GEMM e-graph sizes: LoopLibrarian and SpEQ | 2–10 minutes |
| [Ginsbach/IDL](#ginsbachidl) | IDL detections and output comparisons | 10–30 minutes |
| [KernelFaRer](#kernelfarer-table-9) | Table 9: four complete PolyBench programs | 30–90 minutes |
| [PVA](#pva-table-7) | Table 7: supported operations and correctness | 6–13 minutes |
| [Compilation cost](#compilation-cost-table-8) | Table 8: compilation measurements | 30–60 hours, estimated |
| [Library inventory](#library-inventory-table-1) | Table 1: library specification counts | Under 2 minutes |

## Testing systems

The reproduction uses an x86 Linux build host and a Jetson AGX Orin execution
target. [system-specifications.json](docs/system-specifications.json) contains
the complete machine and library records.

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

| Software | Version used |
|---|---|
| Polygeist | Pinned `third_party/Polygeist` commit |
| LLVM/MLIR/Clang | LLVM 18 revision `26eb4285b56edd8c897642078d91f16ff0fd3472` with the [included parser patch](dependency_sources/llvm-switch-operands.patch) |
| Host tools | Clang 14.0.0; AArch64 GCC/G++ 11.4.0; Python 3.10.12 |
| Python libraries | egglog 11.4.0; NumPy 1.21.5 |
| CUDA | Toolkit 12.6, nvcc 12.6.85; target runtime 12.6.77 |
| Library reference headers | [Pinned public packages](dependency_sources/cuda-headers.json) for source imports and API checks |
| cuBLAS/cuBLASLt | 12.6.1 on Orin; 12.6.4.1 in the cross-compilation SDK |
| Other CUDA libraries | cuSPARSE 12.5.4.2; cuSOLVER 11.7.1.2 |
| Tensor libraries | cuDNN 9.25.1; cuTENSOR 2.7.0 |
| Native ATen comparison | CUDA libtorch `2.6.0a0+df5bbc09d1.nv24.12` |
| OpenBLAS on Orin | 0.3.20, armv8, one thread |
| Other dependencies | Revisions in [repos.json](dependency_sources/repos.json); SpEQ uses its own Docker image |

## Setup and smoke test

Choose Docker for the prebuilt x86 compiler, or install the tools natively.
Both use the same benchmark drivers.

```
git clone https://github.com/PRONTOLab/LoopLibrarian-Benchmarks.git
cd LoopLibrarian-Benchmarks
```

### Docker setup

Use x86-64 Linux with Docker, at least 32 GB RAM, and about 100 GB free disk space.
The image includes LLVM, MLIR, Polygeist, OpenBLAS, Python dependencies,
AArch64 cross-compilers, imported library specifications, and plotting tools.
Pull the image built from artifact revision `ed470507e472` and give it the local
name used by the runner:

```
docker pull ghcr.io/prontolab/looplibrarian-benchmarks:artifact-ed470507e472
docker tag ghcr.io/prontolab/looplibrarian-benchmarks:artifact-ed470507e472 looplibrarian:artifact
bash docker/run.sh polybench_cpu --cases gemm --target host --size MINI --output results/runs/smoke
```

The smoke test takes approximately 1–5 minutes and compares every output of
native and raised GEMM. It checks installation, not Figure 3 performance.
If the registry requires authentication, use an account with package access or
[build the image locally](#building-and-publishing-docker-images).

`docker/run.sh` accepts the same suites and options as `reproduce.py`. In the
experiment commands below, replace `python3 reproduce.py` with `bash docker/run.sh`
to use Docker. Results stay in the host checkout; installed tools remain under
`/opt/looplibrarian`. The `looplibrarian-builds` volume retains additional builds.
Use `bash docker/run.sh shell` for an interactive shell. To select another image,
set `LL_DOCKER_IMAGE`. Docker runs as root; if necessary, restore result ownership:

```
sudo chown -R "$(id -u):$(id -g)" results/runs
```

These are x86-64 build images. They do not execute AArch64 programs or emulate
Orin's GPU or PVA; follow [Execution on Orin](#execution-on-orin) for those runs.

### Native setup

Install Python 3.10 or later, `egglog==11.4.0`, NumPy, CMake, Ninja, Git, GCC/G++,
Clang/LLVM 14, OpenBLAS development files, `file`, and `zstd`. ATen plots also
require `pdflatex` with PGFPlots and `standalone`, plus `pdftoppm`. The root
[Dockerfile](Dockerfile) lists the Ubuntu build packages.

For Orin builds, also install AArch64 GCC/G++, an AArch64 OpenBLAS sysroot,
CUDA 12.6 with nvcc, CCCL headers and the SBSA target, and target-compatible
cuBLAS, cuSOLVER, cuDNN, and cuTENSOR. ATen requires CUDA-enabled AArch64 LibTorch;
PVA requires its separate SDK. Configure their paths before running GPU experiments.

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

### Comparison tools

For Docker, pull the extended image once. The runner selects it automatically
for KernelFaRer, Ginsbach/IDL, equality saturation, and graph sizes:

```
docker pull ghcr.io/prontolab/looplibrarian-benchmarks:comparisons-ed470507e472
docker tag ghcr.io/prontolab/looplibrarian-benchmarks:comparisons-ed470507e472 looplibrarian:comparisons
```

For a native installation, build only the tools needed for your experiment:

```
python3 setup.py kernelfarer
python3 setup.py ginsbach
```

These use the original projects and the included compatibility patches.
Ginsbach/IDL builds its own LLVM fork; allow about 30 minutes with four build jobs.

SpEQ also needs its published reference sources and Docker, including for native
installations. Run the following once before equality-saturation or graph-size
experiments (or `python3 setup.py speq` natively):

```
bash docker/run.sh setup speq
```

This downloads the [published SpEQ artifact](https://zenodo.org/records/10802555)
(6.3 GB), verifies and extracts the needed sources into `third_party/speq`, and
builds `speq-lleq:extract`. Allow about 1–2 hours for a fresh build. The setup
command invokes a build even if a prebuilt image is present. The published image
is also available as `ghcr.io/prontolab/looplibrarian-benchmarks:speq-ab0a3de1345a`.
SpEQ retains its recognizer, library, LLVM passes, and 5/1/3 saturation schedule.
The Docker runner mounts the host Docker socket for SpEQ setup and the two
comparison experiments, granting those containers access to the host daemon.

## Execution on Orin

Execute programs locally on your own compatible device. Set `library_path` in
`config.json` to its CUDA, cuDNN, cuTENSOR, and other shared-library directories.
Configure power mode and clocks before timing, and use an otherwise idle device.
Preflight records clocks, load, temperature, power, and fan state without changing
settings. The tested target uses automatic fan control; the paper used a fixed fan
setting. Record this and any other hardware or input differences with your results.

The x86 image can cross-compile with `--stage build`. Supply `config.docker.json`
with paths as seen inside the container, such as `/sdk/cuda/bin/nvcc`,
`/sdk/cuda/targets/sbsa-linux`, `/sdk/cutensor`, and `/sdk/openblas`:

```
cp config.example.json config.docker.json
# Edit config.docker.json to use the paths inside /sdk.
export LL_DOCKER_CONFIG="$PWD/config.docker.json"
export LL_DOCKER_SDK_ROOT=/path/to/installed-sdks
bash docker/run.sh polybench_gpu --stage build --output results/runs/fig4
```

SDKs are mounted read-only and excluded from the image. Package the built
experiment on the build host:

```
python3 target.py pack --build polybench_gpu=results/runs/fig4 --output results/bundles/gpu
```

Copy this bundle to any directory on Orin, then run it there:

```
python3 target.py run --bundle . --library-path /usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
```

The bundle contains executables, input sources, validation scripts, and unchanged
build records. It needs Python 3.10 or later and the target runtime libraries;
Git and the compiler are not needed. `--verify-only` checks the bundle's hashes
on either machine. Outputs and logs are written to `runs/` inside the bundle.
The original build paths remain in provenance records but are not required on
the target. Existing output directories are not overwritten.

Alternatively, transfer and execute on a machine you control through an existing
SSH configuration. This command also retrieves the results if an experiment fails:

```
python3 target.py remote --bundle results/bundles/gpu --host user@orin --results results/runs/orin-gpu
```

Use `--jump-host user@gateway` if needed and `--library-path` for additional
target libraries. On Orin, select the runtime libraries supplied for its JetPack
installation, including cuBLAS; the cross-compilation SDK is for building only.
Authentication uses your SSH keys or agent; no host or
credentials are included in the artifact. If you use a local `SSH_ASKPASS` helper,
pass `--ssh-option BatchMode=no`; other OpenSSH settings can be supplied with
repeated `--ssh-option KEY=VALUE` arguments. No access to the authors' machines is
needed or provided. A native AArch64 compiler build and AArch64 Docker image have
not been validated.

Repeat `--build SUITE=PATH` to include `llama`, `mfem`, or the `aten` paper
selection in one bundle. Llama also needs `--ggml-build PATH` when packaging,
which copies the four built ggml shared libraries. When packaging from Docker,
that path is `/opt/looplibrarian/build/ggml-aarch64`. Vendor SDKs and shared
libraries remain separate installations on the target. Experiments run serially;
`--suites polybench_gpu llama` selects part of a bundle. Each suite retains its
correctness checks and timing protocol, including five measured samples where
the experiment measures performance.

Additional target configuration:

- **Llama:** set `environment.POLYGEIST_GGML_BUILD` and run `python3 setup.py ggml`
  to build the pinned CUDA baseline. In Docker, use
  `/opt/looplibrarian/build/ggml-aarch64` and `bash docker/run.sh setup ggml`.
- **ATen:** set `environment.ATEN_NATIVE_TORCH_ROOT`, `ATEN_NATIVE_CUDA_ROOT`, and
  `POLYGEIST_CUTENSOR_ROOT` to the target-compatible installations. LibTorch is
  supplied separately. Jiterator tests also need CUDA 12.6 NVRTC and matching
  `libnvrtc-builtins` in the target's `library_path`.
- **PVA:** install the SDK and operator libraries separately. Set
  `environment.PVA_OPERATOR_INCLUDE`, `NVCV_INCLUDE`, `CUPVA_SDK_ROOT`, and
  `PVA_LIB_STAGE` to their installed paths. SDK sources and headers are not
  redistributed here.

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
the timing table. Select a case with `--cases gemm`. `--target host --size MINI`
is a smoke test rather than a Figure 3 measurement.

### PolyBench GPU (Figure 4)

```
python3 reproduce.py polybench_gpu --output results/runs/fig4
```

Run 12 FP64 LARGE kernels on Orin. Five warmups precede five measurements; report
the minimum CUDA-event compute time, excluding allocation and initial/final
transfers. Synchronized end-to-end time is recorded separately. Every output
must pass `atol=rtol=1e-8` before timing is accepted.

`results.csv` contains timings and speedups; per-case files retain checks and a
separate vendor-call trace. These are separate-kernel harnesses: ten native bodies
come from pinned PolyBenchGPU, while correlation and covariance use the paper's
CUDA adaptations. Select a subset with `--cases gemm`.

### Llama

After building the ggml baseline as described under [Execution on Orin](#execution-on-orin):

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

### ATen (Table 5 and Figure 5)

The inputs are 598 extracted/transliterated ATen C kernels in `aten/inputs`, with
extraction records in `aten/metadata`. For Figure 5's 115 cases, using dimensions
from [paper_shapes.json](aten/paper_shapes.json):

```
python3 reproduce.py aten --cohort paper --timing-correctness publication --output results/runs/aten-paper
```

This matches, builds, checks outputs, collects five alternating synchronized
LoopLibrarian/ATen CUDA timing pairs after warmup, and plots the validated results.
Use `--cases aten_mm` for one case (about 2–10 minutes). Options and published
measurements are in [matching_options.json](aten/matching_options.json) and
[paper_timings.csv](aten/paper_timings.csv).

For matching across all 598 inputs at their source dimensions:

```
python3 reproduce.py aten --cohort all --stage match --output results/runs/aten-all
python3 reproduce.py aten --cohort all --stage build --output results/runs/aten-all
python3 reproduce.py aten --cohort all --stage run --output results/runs/aten-all
```

Matching takes an estimated 1–3 hours; supported GPU runs add 1–4 hours. Native
comparison code exists for 170 inputs. Reports identify incomplete matches and
unsupported execution; these dimensions differ from Figure 5. COO-to-CSR still
requires unsupported loop proofs. `--jobs` controls compilation parallelism;
GPU timings are collected serially.

Outputs include `matching/` (IR and proofs), `gpu/` (builds, checks, raw samples),
`gpu/report.csv` (every paper case, including failures), and `summary.json`.
Full-set runs also produce `summary.csv`. The Figure 5 plot is
`plots/aten-current.pdf` and `.png`; its CSV records cases without valid timings.
To regenerate reports or plots without rerunning experiments:

```
python3 reproduce.py aten --cohort all --stage report --output results/runs/aten-all
python3 reproduce.py aten --cohort paper --stage plot --output results/runs/aten-paper
```

Plotting takes seconds. Add `--paper-figure path/to/figure.pdf` for a side-by-side
comparison. The horizontal axis is native CUDA time; the vertical axis is
LoopLibrarian GPU time, in microseconds. The diagonal marks equal time; the
other reference lines mark a factor of two.

**Checks and comparison limits.** The paper command uses FP32 tolerance
`0.002 * (1 + abs(reference))`. A stricter check, `1e-5 + 1e-4 * abs(reference)`,
also distinguishes signed zeros; omitting `--timing-correctness publication`
requires this check before timing. Both modes check full outputs, unchanged
inputs, and buffer guards. Some reductions and convolution gradients agree with
ATen CUDA while failing the stricter serial-C comparison.

New results use the minimum of five samples per implementation. The paper used
five samples for 58 cases and twenty for 57; [paper_timings.csv](aten/paper_timings.csv)
records each protocol. Argmax/argmin use a separately checked finite-input variant
because ATen CUDA differs on the original NaN inputs. Sampled sparse addmm uses
identical column indices for both implementations; its paper timings used different
indices. Reports identify these differences, which limit direct speedup comparisons.

#### Additional ATen pointwise experiments

[pointwise_candidates.csv](aten/pointwise_candidates.csv) lists 268 candidates,
including 54 Figure 5 cases; inclusion does not imply a successful match. Run:

```
python3 aten/pointwise.py --cases exp2 expm1 erfc --output results/runs/pointwise
```

Omit `--cases` for the full inventory. Use `--stage match`, `build`, `run`, or
`report` to resume stages. Matching takes about 20–60 minutes for the inventory,
plus compilation and GPU time for executable cases. `summary.json` separates
candidates, complete matches, strict GPU passes, and timings.

A complete match requires imported-reference proof, valid API arguments, and no
remaining computation loops. The default `--numeric-model strict` can be changed
to `relaxed`, which permits finite-arithmetic transformations such as reassociation
and disregard of signed zero; it does not prove full IEEE equivalence. Use separate
outputs for the two models. Execution checks every output against C and ATen CUDA,
including exceptional values, signed zeros, unchanged inputs, and guards. Five
timing pairs are collected only after strict checks pass. For example, C `roundf`
and ATen round differ on ties and cannot be substituted for one another.

These runs use source dimensions, not Figure 5 dimensions. Optional larger shapes
for 166 candidates are in [pointwise_shapes.json](aten/pointwise_shapes.json):

```
python3 aten/pointwise.py --shape-specs aten/pointwise_shapes.json --cases exp2 expm1 erfc --output results/runs/pointwise-large
```

To combine disjoint runs, pass another run's `gpu/gpu-results.json` with
`--stage report --extra-gpu-results PATH`. The report requires compatible inputs
and compiler settings and rejects duplicates; different dimensions remain separate.

### MFEM

```
python3 reproduce.py mfem --output results/runs/mfem
```

Run the 18 supported contraction fixtures (`NE=1024`) and integrated PCG updates
(`NE=2`) on Orin. Every output is compared with the original computation using
`1e-11 + 1e-10 * abs(expected)`. `summary.json` aggregates results; each case has
`result.json` and `run.log`. This validates execution, without a timing comparison.
The emitted cuBLAS call count does not reproduce the paper's count of 128
cuTensorNet contraction sites.

### Equality saturation

Complete [SpEQ setup](#comparison-tools), then run:

```
python3 reproduce.py eqsat --output results/runs/eqsat
```

Compile all 30 PolyBench programs with equality saturation enabled and disabled,
then compare recognition with SpEQ on the x86 host. `summary.json` separates
recognition from executable lowering. The paper reports 17 recognized programs
with equality saturation and 8 without. The current compiler recognizes 17 and 11
in that set; additional programs outside the set are recorded separately.
SpEQ's reported single match uses GEMM with linearized indexing; results on
unmodified PolyBench are also recorded.

### GEMM e-graph sizes

After [SpEQ setup](#comparison-tools):

```
python3 reproduce.py graph_sizes --output results/runs/graphs
```

`summary.json` reports the counts, with serialized LoopLibrarian graphs and SpEQ
DOT files beside it. The reported result is 370 SpEQ nodes versus 21 LoopLibrarian
nodes. These count different scopes: LoopLibrarian's largest accepted scalar proof
has 18 function nodes and three primitive leaves; SpEQ counts a whole LLVM loop
graph, with primitive values inline in node labels.

### Ginsbach/IDL

After [comparison-tool setup](#comparison-tools):

```
python3 reproduce.py ginsbach --output results/runs/idl
```

Run IDL on all 30 PolyBench programs and execute programs containing replacements
on x86. Expect four detections: one in correlation, two in covariance, and one in
trmm. All three programs fail the full-output comparison; reproducing those
failures is the reported result. `summary.json` records detections and checks.
The driver uses the original allocation helpers in
`inputs/polybench_original_utilities`; changing them affects IDL recognition.

### KernelFaRer (Table 9)

After [comparison-tool setup](#comparison-tools):

```
python3 reproduce.py kernelfarer --output results/runs/table9
```

Compare native compilation, KernelFaRer, and LoopLibrarian for complete FP32 LARGE
GEMM, SYR2K, 2mm, and 3mm programs on x86. Every output must pass
`atol=0.011, rtol=0.0005`, following the paper's two-decimal output convention.
Each implementation has five warmups and five measurements; report the minimum.
`summary.csv` contains comparisons; `timing.json` retains samples.

In that program order, the paper reports LoopLibrarian speedups of 8.6/366/115/93
and KernelFaRer speedups of 8.6/346/113/112. Its native baseline used one warmup
and ten separate process runs, differing from this driver's common five-sample
protocol. Host hardware and load also affect the comparison.

### PVA (Table 7)

After installing the SDK and configuring [target paths](#execution-on-orin):

```
python3 reproduce.py pva --output results/runs/pva
```

Raise public C fixtures, match imported OpenCV references, emit typed PVA calls,
and compare full outputs. This is a correctness experiment, without PVA timings.
The eight cases cover dilation for four integer types, 8-bit-input histograms with
signed/unsigned 32-bit outputs, 8-bit 3×3 Gaussian filtering, and 8-bit 3×3 box
filtering. Histograms and filters use 192×256 inputs. Select a case with
`--cases morphology_u8`; resume execution with `--stage run`.

`summary.json` records results. Per-case files include raised IR, matching proof,
API configuration, and checks. Gaussian and box filters test original, random,
impulse, constant, and quantization-edge inputs, including borders, input
preservation, and guards. Gaussian requires exact equality; its API parameters
come from the imported coefficients. BoxFilter permits a recorded quantization
error bound of 3 over 8-bit pixels. Build records retain the coefficient/rounding
model and accepted bound; results distinguish exact from bounded agreement.

BilateralFilter and HistogramEqualization are unsupported because their library
semantics differ from the fixtures. Other Gaussian pixel types and 16-bit-input
histograms are also outside this driver. These eight cases do not establish
reproduction of every Table 7 entry.

### Compilation cost (Table 8)

```
python3 reproduce.py compilation_cost --output results/runs/compilation-cost
```

[inputs.json](compilation_cost/inputs.json) lists 199 inputs: 30 PolyBench,
23 Llama (19 operation fixtures, extended forward, and three llama2.c functions),
115 ATen, and 31 MFEM. Measure C-to-linked-AArch64 compilation sequentially with
and without equality-saturation rewrites, five times per input: 1,990 builds.
The 30–60-hour estimate comes from initial 50–170-second probes. A small check is:

```
python3 reproduce.py compilation_cost --cases aten/aten_avg_pool2d --repetitions 1 --output results/runs/compilation-check
```

`compilation-table.csv` reports median build times by suite; `measurement-tables.csv`
contains stage times, proof measurements, and peak memory. `summary.json` retains
raw repetitions and failures. A case enters the table only if all repetitions
link successfully; exclusions are reported. Proofs have a ten-second deadline;
timeouts cannot authorize replacements.

The paper used three repetitions; `first-three-compilation-table.csv` reports
repetitions 1–3. Agreement with the paper's measurement boundary and input set
remains unverified. These runs include runtime-object generation and linking.
Application compilation is repeated; parsed reference catalogs are cached per
configuration. Peak RSS is the maximum for one process, not concurrent workers' sum.

To resume, use the suite driver directly with unchanged inputs and compiler:

```
python3 compilation_cost/reproduce.py --resume --output results/runs/compilation-cost
```

Use `python3 reproduce.py compilation_cost --stage summarize --output results/runs/compilation-cost`
to report available data without more builds.

### Library inventory (Table 1)

```
python3 reproduce.py library_inventory --output results/runs/inventory
```

This host-only command writes `inventory.json`, grouping current specifications
by library and comparing the total with the paper's 257. Additional imports in
the current catalog are included in the count.

## Extending the experiments

Use a new output directory and record changes to dimensions, dtypes, or hardware.
For PolyBench, select `--cases gemm --size SMALL`; supported sizes are `MINI`,
`SMALL`, `MEDIUM`, and `LARGE`. Paper Figure 3 uses FP64 LARGE inputs on Orin.

For ATen dimensions, copy [paper_shapes.json](aten/paper_shapes.json) and update
`dims` and related aliases consistently. Pass it through `--shape-specs` to both
[aten_paper115_campaign.py](scripts/correctness/aten_paper115_campaign.py) and
[aten_fixture_batch.py](scripts/correctness/aten_fixture_batch.py); the top-level
ATen command does not expose this option. [aten/reproduce.py](aten/reproduce.py)
shows the invocations. Additional pointwise runs accept `--shape-specs` directly.

To add a benchmark:

1. Add C source, origin, license, dimensions, dtype, and frontend adaptations to
   its experiment directory.
2. Add a harness checking every output against the original. For timings, call
   the real baseline implementation using the same timing boundary.
3. Register the case in the driver. ATen needs a checklist following
   [all_cases.csv](aten/all_cases.csv), shape/argument metadata, and a native
   ATen invocation; both lower-level drivers accept `--checklist`.
4. Check compilation, generated library calls, and output correctness before timing.

To add a library reference in the Polygeist submodule:

1. Add it under `generic_solver/lib_v3/ref_c/` with a manifest under
   `generic_solver/lib_v3/manifest/`. Record public source, argument types/layouts,
   numerical assumptions, and target API. See the [Thrust fill reference](third_party/Polygeist/generic_solver/lib_v3/ref_c/thrust_fill.c)
   and [manifest](third_party/Polygeist/generic_solver/lib_v3/manifest/thrust_fill.json).
2. Add a runtime adapter if needed, calling the library with the declared bindings.
3. Run `python3 setup.py specs` to raise and validate it. Callers and references
   use the same raising and equality-saturation pipeline; no operation-specific
   matcher is required.
4. Add execution checks for supported shapes, layouts, and numerical assumptions;
   reject unsupported inputs. Commit compiler extensions in Polygeist and update
   this repository's submodule pin when distributing them.

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

## Building and publishing Docker images

Build locally instead of pulling if you change the compiler or need another image
revision. Builds use committed sources and exclude local configuration and results:

```
bash docker/build.sh
bash docker/build.sh --comparisons
```

Allow 1–4 hours for a first main-image build. LLVM and Polygeist took about
48 minutes with `BUILD_JOBS=8` on the listed host; lower `BUILD_JOBS` on smaller
machines. The image imports specifications and runs script tests. The comparison
build adds IDL and KernelFaRer with `COMPARISON_JOBS=4` by default. Later builds
reuse unchanged compiler layers. SpEQ is built through [its setup command](#comparison-tools).

For maintainers publishing updated images, commit and build the matching sources,
log Docker into the registry, then run:

```
bash docker/publish.sh
```

This checks revision labels and pushes `artifact` and `comparisons`, plus tags
containing the source revision, to `ghcr.io/prontolab/looplibrarian-benchmarks`.
It does not publish SpEQ. A different registry/repository can be passed as the
first argument. To allow reviewers to pull without authentication, the GitHub
container package must have public visibility.
