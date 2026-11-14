#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
REGISTRY=${1:-ghcr.io/prontolab/looplibrarian-benchmarks}
REVISION=$(git rev-parse HEAD)
if ! git diff --quiet HEAD -- .; then
    echo 'Commit source changes before publishing images.' >&2
    exit 1
fi
# Check both images before publishing either one.
for TARGET in artifact comparisons; do
    BUILT=$(docker image inspect "looplibrarian:$TARGET" \
        --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')
    if [[ "$BUILT" != "$REVISION" ]]; then
        echo "Rebuild looplibrarian:$TARGET for revision $REVISION before publishing." >&2
        exit 1
    fi
done
for TARGET in artifact comparisons; do
    VERSION="$REGISTRY:$TARGET-${REVISION:0:12}"
    docker tag "looplibrarian:$TARGET" "$VERSION"
    docker push "$VERSION"
    docker tag "looplibrarian:$TARGET" "$REGISTRY:$TARGET"
    docker push "$REGISTRY:$TARGET"
done
