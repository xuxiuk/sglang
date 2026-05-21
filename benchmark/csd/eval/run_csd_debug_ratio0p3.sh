#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RUN_RATIOS="0.3"
export CUDA_DEVICES=${CUDA_DEVICES:-2,3}
export CALIB_PORT=${CALIB_PORT:-30003}
export RUN_ROOT=${RUN_ROOT:-/home/zhouxuwen/sglang-debug/benchmark/csd/runs/debug_ratio0p3}
exec bash "${SCRIPT_DIR}/run_csd_three_table_debug.sh"
