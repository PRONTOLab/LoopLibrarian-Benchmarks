#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TARGET=artifact
DEFAULT_IMAGE=looplibrarian:artifact
if [[ ${1:-} == --comparisons ]]; then
    TARGET=comparisons
    DEFAULT_IMAGE=looplibrarian:comparisons
    shift
fi
REVISION=$(git rev-parse HEAD)
COMPILER_REVISION=$(git ls-tree HEAD third_party/Polygeist | awk '{print $3}')
exec docker build --target "$TARGET" \
    --build-arg "ARTIFACT_REVISION=$REVISION" \
    --build-arg "POLYGEIST_REVISION=$COMPILER_REVISION" \
    --build-arg "BUILD_JOBS=${BUILD_JOBS:-8}" \
    --build-arg "COMPARISON_JOBS=${COMPARISON_JOBS:-4}" \
    -t "${LL_DOCKER_IMAGE:-$DEFAULT_IMAGE}" "$@" .
