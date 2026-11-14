# LoopLibrarian artifact

This repository contains the benchmarks and reproduction scripts for
**LoopLibrarian: Automatically Targeting Library Functions as a Compiler ISA**
(CGO ’27). LoopLibrarian raises C computations to MLIR and matches them against
library specifications to generate library calls. The compiler is pinned under
`third_party/Polygeist`.

## Experiments

Run commands from the repository root after completing [setup](docs/reproduction.md).
Use `python3 reproduce.py NAME --output results/runs/NAME`, replacing `NAME`
with an experiment below. Each linked guide describes its inputs, comparison,
and output files.

| Experiment | Paper result or comparison | Approximate time after setup |
|---|---|---|
| [polybench_cpu](polybench/cpu/README.md) | Figure 3: 17 programs, OpenBLAS on Orin CPU | 40 minutes–2 hours |
| [polybench_gpu](polybench/gpu/README.md) | Figure 4: 12 kernels, CUDA on Orin | 20–50 minutes |
| [llama](llama/README.md) | 19-operation coverage and one-layer inference versus ggml | 15–40 minutes |
| [aten](aten/README.md) | Table 5 coverage; Figure 5's 115 timing cases | 1–3 hours for the 115 cases |
| [mfem](mfem/README.md) | Contraction fixtures and PCG updates | 15–45 minutes |
| [eqsat](eqsat/README.md) | Matching with/without equality saturation, versus SpEQ | 30 minutes–2 hours |
| [graph_sizes](graph_sizes/README.md) | GEMM e-graph sizes: LoopLibrarian and SpEQ | 2–10 minutes |
| [ginsbach](ginsbach/README.md) | IDL detections and output comparisons | 10–30 minutes |
| [kernelfarer](kernelfarer/README.md) | Table 9: four complete PolyBench programs | 30–90 minutes |
| [pva](pva/README.md) | Table 7: supported operations and correctness | 6–13 minutes |
| [compilation_cost](compilation_cost/README.md) | Table 8: compilation measurements | 30–60 hours, estimated |
| [library_inventory](library_inventory/README.md) | Table 1: library specification counts | Under 2 minutes |

Times include benchmark compilation and execution on the systems below, but
exclude dependency installation. They are estimates and depend on machine load.
Run GPU experiments one at a time. Missing matches, failed checks, and differences
from the published measurements are reported by the drivers; the per-experiment
guides explain known limits.

## Testing systems

The reproduction uses an x86 Linux build host and a Jetson AGX Orin execution
target. These are the tested versions; [system-specifications.json](docs/system-specifications.json)
contains the complete machine and library records.

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
| LLVM/MLIR/Clang | LLVM 18 development revision `26eb4285b56edd8c897642078d91f16ff0fd3472` |
| Host tools | Clang 14.0.0; AArch64 GCC/G++ 11.4.0; Python 3.10.12 |
| Python libraries | egglog 11.4.0; NumPy 1.21.5 |
| CUDA | Toolkit 12.6, nvcc 12.6.85; target runtime 12.6.77 |
| CUDA libraries | cuBLAS/cuBLASLt 12.6.4.1; cuSPARSE 12.5.4.2; cuSOLVER 11.7.1.2 |
| Tensor libraries | cuDNN 9.25.1; cuTENSOR 2.7.0 |
| Native ATen comparison | CUDA libtorch `2.6.0a0+df5bbc09d1.nv24.12` |
| OpenBLAS on Orin | 0.3.20, armv8, one thread |
| Other dependencies | Revisions in `dependency_sources/repos.json`; SpEQ uses its own Docker image |

The preflight checks record clocks, power, temperature, load, and fan state.
The tested target uses automatic fan control; the paper used a fixed fan setting.
Configure the target before timing; the scripts do not change its power settings.

## Getting started

Install the [required tools and SDKs](docs/reproduction.md#dependencies), then:

```
git clone --recurse-submodules https://github.com/PRONTOLab/LoopLibrarian-Benchmarks.git
cd LoopLibrarian-Benchmarks
cp config.example.json config.json
# Edit config.json for the installed SDKs and execution target.
python3 setup.py sources
python3 setup.py compiler
python3 setup.py specs
```

A clean compiler build takes roughly 4–12 hours; source downloads take
15–60 minutes and specification import takes 2–10 minutes. Llama and the
comparison tools need the additional setup commands in the
[setup guide](docs/reproduction.md#comparison-tools).

Check the installation with one small CPU case (approximately 1–5 minutes):

```
python3 reproduce.py polybench_cpu --cases gemm --target host --size MINI --output results/runs/smoke
```

Then run Figure 3, or the ATen paper experiment and plot:

```
python3 reproduce.py polybench_cpu --output results/runs/fig3
python3 reproduce.py aten --cohort paper --timing-correctness publication --output results/runs/aten-paper
```

For all 598 ATen inputs, use `--cohort all`; see the [ATen guide](aten/README.md)
for matching-only runs, selected cases, and GPU requirements.

## Files and outputs

- Benchmark directories contain C inputs, harnesses, experiment settings, and a `reproduce.py` driver.
- `third_party/` contains the pinned compiler and downloaded dependencies.
- `scripts/correctness/` contains shared harness and comparison code.
- `scripts/artifact/` contains configuration, logging, and local/SSH execution support.
- `results/runs/` receives generated IR, logs, output comparisons, timing samples, and summaries.
- `docs/` contains setup, system details, source notices, and an [extension guide](docs/extending-experiments.md).

Use a new output directory for each build. `--stage plan` prints a command;
`--stage build` compiles; `--stage run --output PATH` executes an existing build.
See each experiment's README for additional stages and result filenames.
Results include source and executable hashes so that measurements can be traced
to the tested inputs. [Source notices](docs/SOURCE_NOTICES.md) describe licensing.
