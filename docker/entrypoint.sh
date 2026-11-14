#!/usr/bin/env bash
set -euo pipefail
# Shared compiler helpers expect these modules on the Python path.
source /opt/looplibrarian/scripts/correctness/common_env.sh
# The checkout is host-owned when mounted by docker/run.sh. Trust this path
# explicitly so Git can record its revision when the container runs as root.
if [[ -e "$PWD/.git" ]]; then
    git config --global --add safe.directory "$PWD"
fi
exec "$@"
