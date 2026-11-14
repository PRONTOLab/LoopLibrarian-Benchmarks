FROM ubuntu:22.04@sha256:74b12b11ffa550fba1929ad4d8d16533c34f6612728469671afba1b03b66881c AS tools
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates clang-14 cmake curl gfortran git graphviz \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libffi-dev libncurses-dev \
    libopenblas-dev libxml2-dev lld-14 llvm-14-dev ninja-build openssh-client \
    pkg-config poppler-utils python3 python3-dev python3-numpy python3-pip \
    python3-venv texlive-latex-extra texlive-pictures time unzip xxd zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir egglog==11.4.0
RUN ln -s /usr/bin/ld.lld-14 /usr/local/bin/ld.lld

FROM tools AS compiler
ARG POLYGEIST_REVISION=7acb7b96299ecc40cbb3ae7ec3fc679db0d15886
ARG BUILD_JOBS=8
ENV POLYGEIST_ROOT=/opt/looplibrarian/third_party/Polygeist \
    POLYGEIST_DEPENDENCY_ROOT=/opt/looplibrarian/third_party \
    POLYGEIST_SPEC_SUITE=lib_v3 \
    PYTHON=/usr/bin/python3 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ENV PATH="${POLYGEIST_ROOT}/build/bin:${POLYGEIST_ROOT}/llvm-project/build/bin:${PATH}"
RUN git init "$POLYGEIST_ROOT" \
    && git -C "$POLYGEIST_ROOT" remote add origin https://github.com/arpitj1/Polygeist.git \
    && git -C "$POLYGEIST_ROOT" fetch --depth 1 origin "$POLYGEIST_REVISION" \
    && git -C "$POLYGEIST_ROOT" checkout --detach FETCH_HEAD
COPY dependency_sources/prepare_llvm.py dependency_sources/llvm.json dependency_sources/llvm-switch-operands.patch /opt/llvm-bootstrap/
RUN python3 /opt/llvm-bootstrap/prepare_llvm.py "$POLYGEIST_ROOT/llvm-project"
RUN cmake -S "$POLYGEIST_ROOT/llvm-project/llvm" -B "$POLYGEIST_ROOT/llvm-project/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS='clang;mlir' \
    -DLLVM_TARGETS_TO_BUILD='X86;ARM;AArch64;NVPTX' -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_LINKER=lld -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
    && cmake --build "$POLYGEIST_ROOT/llvm-project/build" -j "$BUILD_JOBS" \
       --target clang mlir-opt mlir-translate opt llc llvm-link llvm-as llvm-dis llvm-nm
# Polygeist also links the C registration API, outside the command-line targets.
RUN cmake --build "$POLYGEIST_ROOT/llvm-project/build" -j "$BUILD_JOBS" \
    --target MLIRCAPIRegisterEverything
RUN cmake -S "$POLYGEIST_ROOT" -B "$POLYGEIST_ROOT/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DPOLYGEIST_USE_LINKER=lld \
    -DMLIR_DIR="$POLYGEIST_ROOT/llvm-project/build/lib/cmake/mlir" \
    -DCLANG_DIR="$POLYGEIST_ROOT/llvm-project/build/lib/cmake/clang" \
    && cmake --build "$POLYGEIST_ROOT/build" -j "$BUILD_JOBS" --target cgeist polygeist-opt
RUN cmake --build "$POLYGEIST_ROOT/llvm-project/build" -j "$BUILD_JOBS" \
    --target mlir_c_runner_utils mlir_runner_utils

FROM compiler AS prepared
ARG BUILD_JOBS=8
ENV NETLIB_REF=/opt/looplibrarian/third_party/lapack
WORKDIR /opt/looplibrarian
RUN apt-get update && apt-get install -y --no-install-recommends file zstd \
    && rm -rf /var/lib/apt/lists/*
# Keep public-source setup and specification imports cached when only benchmark
# drivers or documentation change.
COPY setup.py config.example.json ./
COPY scripts/artifact/common.py scripts/artifact/common.py
COPY scripts/correctness/ll_paths.py scripts/correctness/ll_paths.py
COPY dependency_sources/repos.json dependency_sources/repos.json
COPY dependency_sources/prepare_cuda_headers.py dependency_sources/cuda-headers.json dependency_sources/
COPY docker/configure.py docker/configure.py
RUN python3 dependency_sources/prepare_cuda_headers.py \
    && python3 docker/configure.py --jobs "$BUILD_JOBS" \
    && (python3 setup.py sources || { cat third_party/upstream-audit.json; tail -80 third_party/setup.log; exit 1; })
RUN python3 setup.py specs || { tail -80 third_party/setup.log; exit 1; }
ARG ARTIFACT_REVISION=8452b2bc8239602d6bf3efadac531614c3311261
LABEL org.opencontainers.image.source="https://github.com/PRONTOLab/LoopLibrarian-Benchmarks" \
      org.opencontainers.image.revision="${ARTIFACT_REVISION}"
# Fetch only the selected revision's Git metadata, then use the local source
# context. Credentials, results, SDKs, and host build trees are excluded.
RUN git init . \
    && git remote add origin https://github.com/PRONTOLab/LoopLibrarian-Benchmarks.git \
    && git fetch --depth 1 origin "$ARTIFACT_REVISION" \
    && git reset --quiet --mixed FETCH_HEAD
COPY . .
FROM prepared AS artifact
RUN python3 -m unittest discover -s tests > /opt/container-tests.log 2>&1 \
    || { tail -80 /opt/container-tests.log; exit 1; }
ENTRYPOINT ["/bin/bash", "/opt/looplibrarian/docker/entrypoint.sh"]
CMD ["bash"]

# Independent toolchains: changing benchmark inputs need not rebuild IDL.
FROM tools AS comparison-tools
ARG COMPARISON_JOBS=4
ENV LL_GINSBACH_ROOT=/opt/comparisons/ginsbach \
    LL_KERNELFARER_ROOT=/opt/comparisons/KernelFaRer \
    LL_JOBS=${COMPARISON_JOBS}
WORKDIR /opt/comparison-setup
COPY setup.py ./
COPY scripts/artifact/common.py scripts/artifact/common.py
COPY scripts/correctness/ll_paths.py scripts/correctness/ll_paths.py
COPY kernelfarer/kernelfarer.patch kernelfarer/kernelfarer.patch
COPY ginsbach/ginsbach-linearalgebra-cuda13.patch ginsbach/ginsbach-linearalgebra-cuda13.patch
RUN (python3 setup.py kernelfarer || { tail -80 third_party/setup.log; exit 1; }) \
    && cmake -S "$LL_KERNELFARER_ROOT" -B /opt/comparisons/kernelfarer-build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-14 -DCMAKE_CXX_COMPILER=clang++-14 \
       -DLLVM_DIR=/usr/lib/llvm-14/lib/cmake/llvm \
    && cmake --build /opt/comparisons/kernelfarer-build -j "$COMPARISON_JOBS"
RUN python3 setup.py ginsbach || { tail -80 third_party/setup.log; exit 1; }

FROM artifact AS comparisons
COPY --from=comparison-tools /opt/comparisons /opt/comparisons
# The client launches the separate SpEQ image through the host Docker daemon.
RUN apt-get update && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*
ENV LL_GINSBACH_ROOT=/opt/comparisons/ginsbach \
    LL_KERNELFARER_ROOT=/opt/comparisons/KernelFaRer
