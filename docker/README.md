# Docker setup

The root `Dockerfile` builds an x86-64 build host with LLVM, MLIR, Polygeist,
OpenBLAS, Python dependencies, AArch64 cross-compilers, public library references,
and plot tools. Use Linux with Docker, at least 32 GB RAM, and about 100 GB free
space. These images are x86-64 build environments. Orin executables run locally
on the reviewer's own AArch64 target; the images do not emulate its GPU or PVA.

## Build and check

From the repository root:

```
bash docker/build.sh
bash docker/run.sh polybench_cpu --cases gemm --target host --size MINI --output results/runs/smoke
```

Allow 1–4 hours for the first build. LLVM and Polygeist compilation took about
48 minutes on the listed build host with `BUILD_JOBS=8` (the default); use fewer
jobs on smaller machines. Later builds reuse the compiler layers. The build imports
library specifications and runs the script tests. The smoke command compiles
and compares native and raised GEMM outputs; it is not a Figure 3 timing run.

`docker/run.sh` accepts the same suites and options as `reproduce.py`. It mounts
this checkout at its original path, saves results on the host, and keeps the
installed tools under `/opt/looplibrarian`. Use `bash docker/run.sh shell` for an
interactive shell. Docker runs as root; generated files may need
`sudo chown -R "$(id -u):$(id -g)" results/runs` afterwards.

## Comparison projects

Build the extended image for KernelFaRer and Ginsbach/IDL. This builds their
pinned sources with the included compatibility patches. IDL has its own LLVM
fork; its build took about 30 minutes with the default `COMPARISON_JOBS=4`.

```
bash docker/build.sh --comparisons
bash docker/run.sh kernelfarer --output results/runs/kernelfarer
bash docker/run.sh ginsbach --output results/runs/ginsbach
```

SpEQ uses a separate image with its own LLVM fork. Its setup downloads the
6.3 GB published archive and builds that image. Run it once before EqSat or
graph-size comparisons:

```
bash docker/run.sh setup speq
bash docker/run.sh graph_sizes --output results/runs/graph_sizes
bash docker/run.sh eqsat --output results/runs/eqsat
```

The helper selects the extended image and mounts the host Docker socket for
these commands. This grants the container access to the host Docker daemon.
SpEQ sources remain in `third_party/speq` on the host.

## Orin experiments

Prepare `config.docker.json` from `config.example.json`. Configure SDK paths
as seen **inside** the container, for example `/sdk/cuda/bin/nvcc`,
`/sdk/cuda/targets/sbsa-linux`, `/sdk/cutensor`, and `/sdk/openblas`.
The last path is the AArch64 OpenBLAS sysroot.

```
export LL_DOCKER_CONFIG="$PWD/config.docker.json"
export LL_DOCKER_SDK_ROOT=/path/to/installed-sdks
bash docker/run.sh polybench_gpu --stage build --output results/runs/polybench_gpu
```

SDKs are mounted read-only and excluded from the build context. The image
downloads pinned public CUDA headers for source imports and API checks;
execution SDKs are supplied separately. Execute the built programs locally on
your own compatible target following the [execution guide](../docs/reproduction.md#execution-on-orin).
The target needs the libraries in the [system specifications](../README.md#testing-systems).
An AArch64 runtime image has not yet been built or tested.

For Llama, set `environment.POLYGEIST_GGML_BUILD` in the configuration to
`/opt/looplibrarian/build/ggml-aarch64`, then run `bash docker/run.sh setup ggml`.
This builds the pinned ggml CUDA baseline with the supplied AArch64 CUDA SDK;
the `looplibrarian-builds` Docker volume retains it across runs.

For ATen, provide the CUDA-enabled AArch64 LibTorch distribution used by the
native baseline and set `environment.ATEN_NATIVE_TORCH_ROOT` and
`environment.ATEN_NATIVE_CUDA_ROOT`. For PVA, also supply the separately installed
SDK described in the [PVA guide](../pva/README.md). These target SDKs and LibTorch
are not built by the host image.

## Publishing images

After committing the sources and building both images, authenticate Docker to
the chosen registry and run:

```
bash docker/publish.sh
```

The default destination is `ghcr.io/prontolab/looplibrarian-benchmarks`. Pass a
different registry/repository as the first argument if needed. The script checks
that both images record the current source revision, then pushes the `artifact`
and `comparisons` tags and tags containing that revision. A successful build
does not publish an image; the registry push must also succeed.
