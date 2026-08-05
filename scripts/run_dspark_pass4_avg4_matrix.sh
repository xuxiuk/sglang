#!/usr/bin/env bash
# Formal DSpark+CSD n=4 matrix on GPU 4-7.
# All four tasks report pass@1 average over four generations and pass@4.
set -euo pipefail

ROOT=${ROOT:-/root/sglang-dspark-csd}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

export ROOT
BASE_RUN_ROOT=${RUN_ROOT:-$ROOT/runs/dspark_csd/pass4_avg4_gpu4_7_$STAMP}
export MODEL=${MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
export CSD_TABLE=${CSD_TABLE:-$ROOT/runs/dspark_csd/v4_mtp314_calibration_dp4_20260802/tables/v4_mtp314_redpajama_merged.json}
export METHODS=${METHODS:-"bare plain dynamic entropy"}
export GPU_SET=4,5,6,7
export PORT=${PORT:-32440}
export DIST_INIT_ADDR=${DIST_INIT_ADDR:-127.0.0.1:32450}
export NCCL_PORT=${NCCL_PORT:-32460}
export NCCL_NET=${NCCL_NET:-Socket}
export MAX_RUNNING_REQUESTS=48
export EVAL_THREADS=48
export MAX_TOKENS=81920
export MAX_MODEL_LENGTH=96000
export ENTROPY_THRESHOLD=${ENTROPY_THRESHOLD:-1.5}
export CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
export LCEVAL_PYTHON=${LCEVAL_PYTHON:-/root/miniconda3/envs/sglang/bin/python}
export LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-$ROOT/runs/mtp_csd/qwen35_accuracy_avg_314_c48_20260802/vendor/lighteval/src}
# The HTTP evaluator must not import the current server source into the older
# LightEval Python environment: its huggingface_hub/transformers versions are
# intentionally frozen. The server still runs from ROOT in its own cu128 env.
export SGLANG_EVAL_SRC=${SGLANG_EVAL_SRC:-/root/sglang/python}
export EVAL_PYTHONPATH="$LIGHTEVAL_SRC:$SGLANG_EVAL_SRC"
export TACO_DATA=${TACO_DATA:-$ROOT/runs/mtp_csd/qwen35_accuracy_avg_314_c48_20260802/vendor/assets/data/taco_1000_code_ood_alpaca.json}
export APPS_DATA=${APPS_DATA:-$ROOT/runs/mtp_csd/qwen35_accuracy_avg_314_c48_20260802/vendor/assets/data/apps_2000_code_ood_alpaca.json}
export CODE_OOD_MAX_EXAMPLES=${CODE_OOD_MAX_EXAMPLES:-1000}
export CODE_OOD_NUM_SAMPLES=${CODE_OOD_NUM_SAMPLES:-4}

test -s "$MODEL/config.json"
test -s "$CSD_TABLE"
test -d "$LIGHTEVAL_SRC/lighteval"
test -s "$TACO_DATA"
test -s "$APPS_DATA"

test ! -e "$BASE_RUN_ROOT" || {
  echo "Refusing to reuse run directory: $BASE_RUN_ROOT" >&2
  exit 1
}
mkdir -p "$BASE_RUN_ROOT"
cat >"$BASE_RUN_ROOT/phase_order.txt" <<EOF
phase1_tasks=aime25_avg4 math500_avg4 lcb_avg4 gsm8k_avg4
phase2_tasks=taco apps
methods=$METHODS
csd_table=$CSD_TABLE
EOF

# Finish every method on the scored tasks before spending time on the two
# code-OOD generation workloads.
TASKS="aime25_avg4 math500_avg4 lcb_avg4 gsm8k_avg4" \
RUN_ROOT="$BASE_RUN_ROOT/main_tasks" \
bash "$ROOT/scripts/run_dspark_accuracy81920_matrix.sh"

TASKS="taco apps" \
RUN_ROOT="$BASE_RUN_ROOT/code_ood" \
bash "$ROOT/scripts/run_dspark_accuracy81920_matrix.sh"
