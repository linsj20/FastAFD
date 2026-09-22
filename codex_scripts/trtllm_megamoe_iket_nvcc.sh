#!/usr/bin/env bash
# Only profiling processes select this wrapper; ordinary compilation is untouched.
set -euo pipefail
exec /usr/local/cuda/bin/nvcc -I"${TRT_SWEEP_ROOT:?}/marked" -I"${TRT_IKET_INCLUDE:?}" \
    -DIKET_BUILD_ENABLED=1 "$@"
