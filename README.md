# LoopLibrarian benchmarks

Reproduction sources, scripts, correctness checks, and reference measurements
for the LoopLibrarian paper. Polygeist is a pinned dependency under
`third_party/Polygeist`; its `artifact-submission` branch contains the compiler,
runtime adapters, imported reference specifications, and compiler tests.

Repository: https://github.com/PRONTOLab/LoopLibrarian-Benchmarks (public).

## Testing systems

The following is the **current reproduction environment**, verified on
2026-11-12. Historical paper measurements retain their own provenance; these
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
are ignored. The `aten/inputs` and `mfem/inputs` links expose the preserved extraction
fixtures; historical output dumps are never used as correctness oracles.
Each successful run retains its commands, source/compiler hashes,
intermediate IR, correctness checks, raw samples, and paper comparison.
See [docs/reproduction.md](docs/reproduction.md) for the original eight evaluation
protocols and [aten/README.md](aten/README.md) for the incomplete ATen cohort.

## Current reproduction status

The eight established drivers are PolyBench CPU/GPU, Llama, MFEM, EqSat, graph
sizes, Ginsbach, and KernelFaRer. ATen currently has 106/115 complete matches and
102 strict GPU correctness passes; four matched cases retain failed comparisons
and nine remain unmatched. PVA's fresh compilation flow, the repeated compilation
cost table, and the specification inventory still require completion. Moving
files between repositories does not change these scientific results.
