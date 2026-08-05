#!/usr/bin/env bash
# Formal DSpark n=4 matrix on GPU 4-7: four tasks x four methods. P20 is the
# historical upper entropy gate: reject the highest-entropy 20%, whose
# calibrated ascending-P80 cutoff is 1.5638477802276611.
set -euo pipefail

ROOT=${ROOT:-/root/sglang-dspark-csd}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

export ROOT
export RUN_ROOT=${RUN_ROOT:-$ROOT/runs/dspark_csd/four_tasks_n4_p20_gpu4_7_$STAMP}
export MODEL=${MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
export CSD_TABLE=${CSD_TABLE:-$ROOT/runs/dspark_csd/v4_mtp314_calibration_dp4_20260802/tables/v4_mtp314_redpajama_merged.json}
export METHODS="bare plain dynamic entropy"
export TASKS="aime25_avg4 math500_avg4 lcb_avg4 gsm8k_avg4"
export CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
export ENTROPY_THRESHOLD=${ENTROPY_THRESHOLD:-1.5638477802276611}
export ENTROPY_MIN_THRESHOLD=-1
export GPU_SET=${GPU_SET:-4,5,6,7}
export PORT=${PORT:-32540}
export DIST_INIT_ADDR=${DIST_INIT_ADDR:-127.0.0.1:32550}
export NCCL_PORT=${NCCL_PORT:-32560}
export NCCL_NET=${NCCL_NET:-Socket}
# LightEval sends 48 concurrent HTTP requests with n=4. SGLang divides this
# global sequence limit over DP=4, hence 192 gives 48 running slots per rank.
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-192}
export EVAL_THREADS=${EVAL_THREADS:-48}
export MAX_TOKENS=${MAX_TOKENS:-81920}
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-96000}
export LCEVAL_PYTHON=${LCEVAL_PYTHON:-/root/miniconda3/envs/sglang/bin/python}
export LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-$ROOT/runs/mtp_csd/qwen35_accuracy_avg_314_c48_20260802/vendor/lighteval/src}
export SGLANG_EVAL_SRC=${SGLANG_EVAL_SRC:-/root/sglang/python}
export EVAL_PYTHONPATH="$LIGHTEVAL_SRC:$SGLANG_EVAL_SRC"

test -s "$MODEL/config.json"
test -s "$CSD_TABLE"

exec bash "$ROOT/scripts/run_dspark_accuracy81920_matrix.sh"
