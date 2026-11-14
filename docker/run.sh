#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:?usage: docker/run.sh SUITE [OPTIONS] | setup COMPONENT | shell}
shift
IMAGE=${LL_DOCKER_IMAGE:-looplibrarian:artifact}
ARGS=(--rm --init -v "$ROOT:$ROOT" -w "$ROOT"
      -v looplibrarian-builds:/opt/looplibrarian/build
      -e LL_ARTIFACT_CONFIG=/opt/looplibrarian/config.json)
if [[ -t 0 && -t 1 ]]; then ARGS+=(-it); fi
if [[ -n ${LL_DOCKER_CONFIG:-} ]]; then
    ARGS+=(-v "$(realpath "$LL_DOCKER_CONFIG"):/opt/looplibrarian/config.json:ro")
fi
if [[ -n ${LL_DOCKER_SDK_ROOT:-} ]]; then
    ARGS+=(-v "$(realpath "$LL_DOCKER_SDK_ROOT"):/sdk:ro")
fi
if [[ -n ${LL_DOCKER_SSH_DIR:-} ]]; then
    ARGS+=(-v "$(realpath "$LL_DOCKER_SSH_DIR"):/root/.ssh:ro")
fi
case "$ACTION" in
    eqsat|graph_sizes|ginsbach|kernelfarer)
        IMAGE=${LL_DOCKER_IMAGE:-looplibrarian:comparisons} ;;
esac
if [[ "$ACTION" == eqsat || "$ACTION" == graph_sizes ||
      ( "$ACTION" == setup && ${1:-} == speq ) ]]; then
    IMAGE=${LL_DOCKER_IMAGE:-looplibrarian:comparisons}
    ARGS+=(-v /var/run/docker.sock:/var/run/docker.sock
           -e "LL_SPEQ_ROOT=$ROOT/third_party/speq")
fi
case "$ACTION" in
    shell) COMMAND=(bash "$@") ;;
    setup)
        ARGS+=(-w /opt/looplibrarian)
        COMMAND=(python3 setup.py "$@") ;;
    *) COMMAND=(python3 reproduce.py "$ACTION" "$@") ;;
esac
exec docker run "${ARGS[@]}" "$IMAGE" "${COMMAND[@]}"
