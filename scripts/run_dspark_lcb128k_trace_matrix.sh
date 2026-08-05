#!/usr/bin/env bash
# Corrected full LCB-v6 accuracy/throughput matrix for DSpark+CSD.
# Uses the official LiveCodeBench prompt, the stdin.buffer-compatible judge,
# 128K Think-High budget, and streaming per-request timing traces.

set -euo pipefail

ROOT=${ROOT:-/root/sglang-dspark-csd}
export ROOT
export RUN_ROOT=${RUN_ROOT:-$ROOT/runs/dspark_csd/lcb128k_trace_matrix_$(date +%Y%m%d_%H%M%S)}
export METHODS=${METHODS:-"bare plain dynamic entropy"}
export TASKS=lcb
export GPU_SET=${GPU_SET:-4,5,6,7}
export PORT=${PORT:-31000}
export DIST_INIT_ADDR=${DIST_INIT_ADDR:-127.0.0.1:21000}
export NCCL_PORT=${NCCL_PORT:-31001}
export NCCL_NET=${NCCL_NET:-Socket}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
export EVAL_THREADS=${EVAL_THREADS:-48}
export MAX_TOKENS=${MAX_TOKENS:-128000}
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-128000}

exec bash "$ROOT/scripts/run_dspark_accuracy81920_matrix.sh"
