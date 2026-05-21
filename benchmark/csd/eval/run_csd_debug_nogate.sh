#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RUN_RATIOS="0"
export CUDA_DEVICES=${CUDA_DEVICES:-0,1}
export CALIB_PORT=${CALIB_PORT:-30002}
export RUN_ROOT=${RUN_ROOT:-/home/zhouxuwen/sglang-debug/benchmark/csd/runs/debug_nogate}
exec bash "${SCRIPT_DIR}/run_csd_three_table_debug.sh"
