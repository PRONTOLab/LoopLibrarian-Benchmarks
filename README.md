# Artifact for "LoopLibrarian: Automatically Targeting Library Functions as a Compiler ISA" (CGO '27)

This repository contains the artifact for the paper:
> **"LoopLibrarian: Automatically Targeting Library Functions as a Compiler ISA"**  
> *IEEE/ACM International Symposium on Code Generation and Optimization (CGO '27)*

Artifact DOI: [![DOI](https://zenodo.org/badge/1455872115.svg)](https://zenodo.org/badge/latestdoi/1455872115)

[![Artifact badge target: Reusable](https://img.shields.io/badge/Artifact_badge_target-Reusable-lightgrey)](docs/extending-experiments.md)

Reproduction sources, scripts, correctness checks, and reference measurements
for the LoopLibrarian paper. Polygeist is a pinned dependency under
`third_party/Polygeist`; its `artifact-submission` branch contains the compiler,
runtime adapters, imported reference specifications, and compiler tests.

Repository: https://github.com/PRONTOLab/LoopLibrarian-Benchmarks (public).

## Reuse and extension

We target the **Artifacts Evaluated — Reusable** badge described in the
[CGO artifact evaluation guidance](https://2027.cgo.org/track/cgo-2027-artifact-evaluation).
The badge above identifies this target; the evaluation badge has not yet been awarded.

Experiments can be extended at several levels: select existing cases or change
PolyBench dataset sizes through command-line options, configure another testing
system, add a C benchmark and its correctness harness, or import a new library
reference through the shared raising and equality-saturation pipeline. A new
executable library backend also needs an API binding and validation.

The [extension guide](docs/extending-experiments.md) gives commands, identifies
the files to change and manual work required, and explains how to retain
correctness and performance evidence. See [expected experiment time](#expected-experiment-time)
for planning estimates after setup.

Submission preparation is tracked in the
[artifact checklist](docs/artifact-submission-checklist.md), using the cTuning
author guidance and CGO 2027 requirements.

## Testing systems

The following is the **current reproduction environment**.
Historical paper measurements retain their own provenance; these
software versions must not be retroactively attributed to the paper.
Machine-readable details are in [docs/system-specifications.json](docs/system-specifications.json).

| Component | Build/comparison host | Execution target |
|---|---|---|
| System | KVM virtual machine | NVIDIA Jetson AGX Orin Developer Kit |
| Architecture | x86-64 | AArch64 |
| CPU | Intel Xeon Processor (Cascade Lake), 24 exposed vCPUs | 12 online CPU cores |
| RAM visible to OS | approximately 62 GiB | 64,387,204 KiB, approximately 61.4 GiB |
| GPU | Not used for target timing | Integrated Orin GPU, CUDA compute capability 8.7 (`sm_87`) |
| Operating system | Ubuntu 22.04.1 LTS | Ubuntu 24.04.4 LTS |
| Linux kernel | 5.15.0-151-generic | 6.8.12-1021-tegra |
| CPU clock policy | Shared VM; record load for every timing | All clusters fixed at 2.2016 GHz |
| GPU clock policy | Not applicable | Minimum and maximum both 1.3005 GHz |
| Power mode | VM host-managed | MAXN, mode 0 |

Figure 3 runs on the **Orin CPU**, pinned to CPU 0 with one OpenBLAS thread.
PolyBench GPU, Llama, ATen, and MFEM GPU execution use the Orin. KernelFaRer
and IDL/Ginsbach host comparisons, EqSat, graph measurements, and compiler
builds use the x86 host. PVA tests use the Orin's PVA through a separately
installed SDK; PVA runtime timings are not a paper reproduction target.

### Software used

| Component | Recorded version/configuration |
|---|---|
| Polygeist | Exact `third_party/Polygeist` submodule commit; branch `artifact-submission` |
| LLVM/MLIR/Clang used by Polygeist | LLVM 18.0.0 development revision `26eb4285b56edd8c897642078d91f16ff0fd3472` |
| System Clang / resource headers | Ubuntu Clang 14.0.0 |
| AArch64 GCC/G++ | Ubuntu 11.4.0 cross toolchain |
| Host Python | `/usr/bin/python3`, Python 3.10.12 |
| Python packages | egglog 11.4.0; NumPy 1.21.5 |
| CUDA compilation toolkit | CUDA 12.6, nvcc 12.6.85; AArch64 SBSA target |
| CUDA runtime on target | libcudart 12.6.77 |
| CUDA driver API reported on target | 13030; this is the driver API value, not the compilation toolkit version |
| cuBLAS / cuBLASLt | 12.6.4.1 available; per-run loaded-library hashes identify the actual selected files |
| cuSPARSE | 12.5.4.2 |
| cuSOLVER | 11.7.1.2 |
| cuDNN | 9.25.1; runtime API reports 92501 |
| cuTENSOR | 2.7.0; runtime API reports 20700 |
| ATen native comparison | Real CUDA libtorch, `2.6.0a0+df5bbc09d1.nv24.12`, built for CUDA 12.6 |
| OpenBLAS | 0.3.20 on Orin; `DYNAMIC_ARCH`, `NO_AFFINITY`, armv8, built with `MAX_THREADS=64`; measurements use one thread |
| Other source dependencies | Pinned submodules and `dependency_sources/repos.json` |
| SpEQ | Published artifact and pinned LLVM fork; separate Docker environment |

CUDA libraries and PVA SDK components are installed separately. They are not
redistributed here. The source importer also needs public reference sources and
vendor API headers. Compiler and benchmark configurations use separate paths;
no developer home directory is required by the main launcher.

### Measurement settings

- Five warmup executions, followed by five measured samples. ATen uses five
  alternating, synchronized raised/native pairs. Retained historical twenty-pair
  results stay explicitly labeled with their original protocol.
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
- Record CPU affinity, actual CPU/GPU clocks, power mode, host load, target load,
  temperature, fan state, compiler hash, input shapes, loaded-library hashes,
  numeric model, and cache settings for every measured campaign.
- The current Orin uses automatic fan control. The paper's fixed-fan condition
  differs and must be disclosed when comparing timings. Capture the actual fan
  state rather than assuming a constant PWM value.
- Preserve each suite's timing boundary: PolyBench GPU's reported compute time
  excludes allocation and initial/final transfers; synchronized end-to-end times
  are retained separately. Library/allocator caches are recorded when enabled.
- PolyBench CPU/GPU use FP64 LARGE inputs. Llama uses the recorded FP32 one-layer,
  one-token fixture. ATen uses the paper shape manifest, not its small source
  defaults, and preserves each fixture's dtype.
- Run full-output correctness checks before accepting timings. The README for
  each suite states its actual comparison and tolerance; failed cases remain
  failed, and timing differences are reported without changing acceptance gates.

## Repository layout

```
third_party/             pinned compiler and source dependencies
polybench/cpu/          Figure 3
polybench/gpu/          Figure 4
polybench/sources/      original PolyBench inputs
llama/                 inference fixtures and expert comparison
aten/                  coverage, correctness, and performance
mfem/                  contraction fixtures and PCG updates
pva/                   coverage and correctness; no timing target
eqsat/                 submitted 17-versus-8-versus-1 comparison
graph_sizes/           GEMM graph-size comparison
kernelfarer/           complete-program comparison
ginsbach/              IDL detection/correctness comparison
compilation_cost/      compilation measurements
library_inventory/     imported-specification inventory
inputs/                shared pinned fixture inputs
scripts/               shared experiment drivers
results/archive/       retained evidence and original extraction fixtures
docs/                  methodology, system metadata, migration record
```

The eight established suites, ATen, and the library inventory have reproduction
entry points. PVA and compilation-cost documentation identifies the remaining
setup and validation work.
The root launcher provides a common interface; pending evaluations are labeled
as pending rather than presented as completed reproduction results.

## Getting started

```
git clone --recurse-submodules git@github.com:PRONTOLab/LoopLibrarian-Benchmarks.git
cd LoopLibrarian-Benchmarks
cp config.example.json config.json
python3 setup.py sources
python3 setup.py compiler
python3 setup.py specs
python3 reproduce.py polybench_cpu --stage plan
python3 reproduce.py polybench_cpu --output results/runs/fig3
```

Configure SDK paths, cross-compilation tools, and an SSH alias in `config.json`.
Keep credentials in SSH/client configuration; `config.json` and fresh run outputs
are ignored. The benchmark directories contain real source files: `aten/inputs/`,
`mfem/inputs/`, `llama/inputs/`, and `polybench/sources/`. ATen cohort and shape
manifests live in `aten/paper_cases.csv` and `aten/paper_shapes.json`; inventories
and provenance live in `aten/metadata/`. Saved results under `results/archive/`
are comparison evidence and are not required to locate these source inputs.
See [the repository layout](docs/repository-layout.md) for source, metadata,
compiler dependency, and result locations.
Each successful run retains its commands, source/compiler hashes,
intermediate IR, correctness checks, raw samples, and paper comparison.
See [docs/reproduction.md](docs/reproduction.md) for the original eight evaluation
protocols and [aten/README.md](aten/README.md) for the incomplete ATen cohort.

## Expected experiment time

Budget **2–10 minutes for one selected ATen case**, or **roughly 3–9 hours for
the eight established evaluation suites**, once their dependencies are built.
The full ATen campaign is additional. A first installation needs substantially
longer because it builds LLVM and the comparison toolchains.

These are **rough wall-clock planning estimates** for the
systems above. They are not kernel runtimes or measured end-to-end completion
times for every driver. They assume working SDK/SSH configuration, available
input files, no competing GPU workload, and the default five warmups and five
measurements (five raised/native pairs for ATen). Shared-host load, compilation
parallelism, cold caches, and network speed can move a run outside these ranges.
Run GPU timing campaigns serially.

### First-time preparation

| Preparation | Rough allowance | Scope |
|---|---|---|
| Clone and fetch public sources | 15–60 minutes | Network-dependent; excludes installing vendor SDKs and the SpEQ archive |
| LLVM/MLIR and Polygeist: `setup.py compiler` | 4–12 hours | Clean build; conservative allowance for the default three build jobs |
| Reference import and validation: `setup.py specs` | 2–10 minutes | Existing compiler and source dependencies; rerun after compiler/specification changes |
| ggml CUDA dependency: `setup.py ggml` | 30 minutes–2 hours | First CUDA cross-compilation for Llama |
| IDL toolchain: `setup.py ginsbach` | 2–6 hours | Separate historical LLVM build |
| SpEQ environment: `setup.py speq` | 2–6 hours, plus download | Separate LLVM/Docker build and a 6.3 GB artifact download |

Allow one or two days for a complete first setup, including toolchain builds
and local configuration. This is a scheduling allowance, not an unattended
installation guarantee. Subsequent experiments reuse these dependencies.

### Experiments after preparation

The following ranges include fresh benchmark compilation/matching, deployment
where needed, correctness checks, execution, and result summarization. They
exclude the preparation above. Use the suite name with `python3 reproduce.py`.

| Experiment | Scope | Rough total time |
|---|---|---|
| Small smoke test | One PolyBench MINI case on the host | 1–5 minutes |
| `polybench_cpu` — Figure 3 | All 17 LARGE cases, baseline and raised | 40 minutes–2 hours |
| `polybench_gpu` — Figure 4 | All 12 LARGE cases, correctness, timing, and vendor-call audit | 20–50 minutes |
| `llama` | 19 operation fixtures and the one-layer, one-token expert comparison | 15–40 minutes |
| `mfem` | 18 supported fixtures and integrated PCG updates | 15–45 minutes |
| `eqsat` | 30 inputs with/without EqSat, plus SpEQ comparisons | 30 minutes–2 hours |
| `graph_sizes` | Paired GEMM graphs for LoopLibrarian and SpEQ | 2–10 minutes |
| `ginsbach` | 30-input detection sweep and correctness of detected replacements | 10–30 minutes |
| `kernelfarer` | GEMM, SYR2K, 2mm, and 3mm; three implementations | 30–90 minutes |
| `library_inventory` | Audit the existing specification catalog | Less than 2 minutes; import is separate |
| `aten`, selected case | Paper shape, correctness, and five timing pairs | 2–10 minutes; large convolution cases can take longer |
| `aten`, full 115-case attempt | Matching, builds, correctness, and timings for passing cases | 2–8 hours; provisional estimate, cohort still incomplete |

For repeat runs using valid existing executables, budget approximately
30–90 minutes for PolyBench CPU, 10–20 minutes for PolyBench GPU, and
2–10 minutes for Llama. Use `--stage run --output <existing-run-directory>`
where supported. Rebuild after changing compiler, sources, shapes, or build
settings; reusing stale executables does not reproduce the new configuration.

The estimates are informed by recent component runs: the recorded
[Figure 4 campaign](results/archive/issues/fig4_gpu_run-ignecjbdac/) contains 38
per-process `elapsed_s` records totaling about eight minutes. That total
excludes compilation and other orchestration, so it is not the full-suite
duration. Recent source imports took roughly two minutes, and selected ATen
build/validation runs took minutes. The full-suite ranges above remain planning
estimates until a complete run of each packaged driver records its duration.
Timeout settings are failure limits, not expected run times.

The 200-pointwise-case ATen target, PVA's fresh compilation flow, and the repeated
compilation-cost table are still under development. Their complete reproduction
times are not yet established; the ATen estimate above does not promise that all
115 cases pass or include time to implement missing compiler support.

Record actual elapsed time when running an experiment, for example:

```
mkdir -p results/runs
/usr/bin/time -p -o results/runs/fig3-wall-time.txt python3 reproduce.py polybench_cpu --output results/runs/fig3
```

Use a fresh output directory for each build. The `real` value in the timing file
is the experiment's wall-clock duration in seconds; keep it with the results
and recorded machine load when refining these estimates.
