#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:?usage: docker/run.sh SUITE [OPTIONS] | setup COMPONENT | shell}
shift
IMAGE=${LL_DOCKER_IMAGE:-looplibrarian:artifact}
ARGS=(--rm --init -v "$ROOT:$ROOT" -w "$ROOT"
      -e LL_ARTIFACT_CONFIG=/opt/looplibrarian/config.json)
if [[ -t 0 && -t 1 ]]; then ARGS+=(-it); fi
if [[ -n ${LL_DOCKER_CONFIG:-} ]]; then
    ARGS+=(-v "$(realpath "$LL_DOCKER_CONFIG"):/opt/looplibrarian/config.json:ro")
fi
if [[ -n ${LL_DOCKER_SDK_ROOT:-} ]]; then
    ARGS+=(-v "$(realpath "$LL_DOCKER_SDK_ROOT"):/sdk:ro")
fi
for ENTRY in PVA_OPERATOR_INCLUDE:operator NVCV_INCLUDE:nvcv CUPVA_SDK_ROOT:cupva PVA_LIB_STAGE:lib; do
    VARIABLE=${ENTRY%%:*}
    MOUNT_NAME=${ENTRY#*:}
    HOST_VARIABLE=LL_DOCKER_$VARIABLE
    if [[ -n ${!HOST_VARIABLE:-} ]]; then
        ARGS+=(-v "$(realpath "${!HOST_VARIABLE}"):/pva/$MOUNT_NAME:ro"
               -e "$VARIABLE=/pva/$MOUNT_NAME")
    fi
done

GPU_TARGET=${LL_GPU_TARGET:-x86_64}
USE_BUILD_VOLUME=1
STAGE=all
COHORT=paper
PREVIOUS=
for ARG in "$@"; do
    if [[ "$PREVIOUS" == --gpu-target ]]; then GPU_TARGET=$ARG; fi
    if [[ "$PREVIOUS" == --stage ]]; then STAGE=$ARG; fi
    if [[ "$PREVIOUS" == --cohort ]]; then COHORT=$ARG; fi
    case "$ARG" in
        --gpu-target=*) GPU_TARGET=${ARG#*=} ;;
        --stage=*) STAGE=${ARG#*=} ;;
        --cohort=*) COHORT=${ARG#*=} ;;
    esac
    PREVIOUS=$ARG
done
case "$GPU_TARGET" in x86_64|jetson) ;; *) echo 'GPU target must be x86_64 or jetson' >&2; exit 2 ;; esac
ARGS+=(-e "LL_GPU_TARGET=$GPU_TARGET")
case "$ACTION" in
    polybench_gpu|llama|mfem|aten|eqsat_extra|compilation_cost)
        if [[ "$GPU_TARGET" == x86_64 ]]; then
            IMAGE=${LL_DOCKER_IMAGE:-looplibrarian:gpu}
            USE_BUILD_VOLUME=0
            NEED_GPU=1
            if [[ "$ACTION" == compilation_cost ]]; then NEED_GPU=0; fi
            case "$STAGE" in plan|match|build|report|summarize|plot) NEED_GPU=0 ;; esac
            if [[ "$ACTION" == aten && ( "$STAGE" == match ||
                  ( "$COHORT" == all && ( "$STAGE" == report || "$STAGE" == summarize ) ) ) ]]; then
                NEED_GPU=0
                IMAGE=${LL_DOCKER_IMAGE:-looplibrarian:artifact}
                USE_BUILD_VOLUME=1
            fi
            if [[ "$NEED_GPU" == 1 ]]; then ARGS+=(--gpus "${LL_DOCKER_GPUS:-all}" --ulimit memlock=-1); fi
        fi ;;
    setup)
        if [[ ${1:-} == ggml && "$GPU_TARGET" == x86_64 ]]; then
            IMAGE=${LL_DOCKER_IMAGE:-looplibrarian:gpu}
            USE_BUILD_VOLUME=0
        fi ;;
esac
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
if [[ "$USE_BUILD_VOLUME" == 1 ]]; then ARGS+=(-v looplibrarian-builds:/opt/looplibrarian/build); fi
case "$ACTION" in
    shell) COMMAND=(bash "$@") ;;
    setup)
        ARGS+=(-w /opt/looplibrarian)
        COMMAND=(python3 setup.py "$@") ;;
    *) COMMAND=(python3 reproduce.py "$ACTION" "$@") ;;
esac
exec docker run "${ARGS[@]}" "$IMAGE" "${COMMAND[@]}"
