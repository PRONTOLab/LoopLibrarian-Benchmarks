#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
IMAGE=${LL_DOCKER_IMAGE:-looplibrarian:artifact}
REGISTRY=${1:-ghcr.io/prontolab/looplibrarian-benchmarks}
REVISION=$(git rev-parse HEAD)
if ! git diff --quiet HEAD -- .; then
    echo 'Commit source changes before publishing images.' >&2
    exit 1
fi
BUILT=$(docker image inspect "$IMAGE" \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')
if [[ "$BUILT" != "$REVISION" ]]; then
    echo "Rebuild $IMAGE for revision $REVISION before publishing." >&2
    exit 1
fi
docker tag "$IMAGE" "$REGISTRY:latest"
docker push "$REGISTRY:latest"
