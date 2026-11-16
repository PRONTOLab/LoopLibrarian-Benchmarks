#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if ! git diff --quiet HEAD -- .; then
    echo 'Commit source changes before building release images.' >&2
    exit 1
fi
# One image now holds what these options used to select.
case ${1:-} in
    --gpu|--comparisons)
        echo "$1 is no longer needed: one image contains the compiler, GPU libraries, and comparison tools." >&2
        shift ;;
esac
REVISION=$(git rev-parse HEAD)
COMPILER_REVISION=$(git ls-tree HEAD third_party/Polygeist | awk '{print $3}')
# Match the image's revision label even when the checkout has untracked files.
BUILD_CONTEXT=$(mktemp -d)
trap 'rm -rf "$BUILD_CONTEXT"' EXIT
git archive HEAD | tar -x -C "$BUILD_CONTEXT"
docker build \
    --build-arg "ARTIFACT_REVISION=$REVISION" \
    --build-arg "POLYGEIST_REVISION=$COMPILER_REVISION" \
    --build-arg "BUILD_JOBS=${BUILD_JOBS:-8}" \
    --build-arg "COMPARISON_JOBS=${COMPARISON_JOBS:-4}" \
    --build-arg "GPU_ARCH=${LL_GPU_ARCH:-80}" \
    -t "${LL_DOCKER_IMAGE:-looplibrarian:artifact}" "$@" "$BUILD_CONTEXT"
