#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export POLYGEIST_ROOT="${POLYGEIST_ROOT:-$root/third_party/Polygeist}"
exec "$POLYGEIST_ROOT/scripts/correctness/cached_nvcc.sh" "$@"
