#!/usr/bin/env bash
# Rerun 515 + calib1024 no-ratio variants after CSD runtime changes:
#   1. plain_entropy_p20: static CSD + P20 entropy gate
#   2. dynamic_ignore_ratio: dynamic update, ignore prob-ratio for pair collection
#   3. dynamic_entropy_p20_ignore_ratio: dynamic no-ratio + P20 entropy gate
#
# This intentionally uses one TP=4 GPU group and leaves CSD_REBUILD_TOP_KEEP
# empty so the rebuild table is not top-k truncated.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="/root/miniconda3/envs/sglang/bin:${CUDA_HOME}/bin:/home/ccuser/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export SGLANG_SRC=${SGLANG_SRC:-"${REPO_ROOT}/python"}
export LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-"${REPO_ROOT}/benchmark/csd/lighteval/src"}
export PYTHONPATH="${SGLANG_SRC}:${LIGHTEVAL_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_515_calib1024_dynamic_p20_no_ratio_rerun}
OUT_ROOT_BASE=${OUT_ROOT_BASE:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_515_calib1024_dynamic_p20_no_ratio_rerun/${RUN_STAMP}"}
MERGED_RESULT_FILE=${MERGED_RESULT_FILE:-"${OUT_ROOT_BASE}/results/dynamic_p20_no_ratio_rerun.jsonl"}
DRIVER_LOG_DIR=${DRIVER_LOG_DIR:-"${OUT_ROOT_BASE}/driver_logs"}

PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/runs/redpajama/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json"}

TREE_SHAPES=${TREE_SHAPES:-"515:5:1:5"}
TASK_FILTER=${TASK_FILTER:-all}
METHOD_SET=${METHOD_SET:-"plain_entropy_p20 dynamic_ignore_ratio dynamic_entropy_p20_ignore_ratio"}

CUDA_DEVICES=${CUDA_DEVICES:-4,5,6,7}
PORT=${PORT:-30014}
TP_SIZE=${TP_SIZE:-4}

MODEL_PATH=${MODEL_PATH:-/root/model/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-7200}
MAMBA_SCHEDULER_STRATEGY=${MAMBA_SCHEDULER_STRATEGY:-no_buffer}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}
SHOW_KERNEL=${SHOW_KERNEL:-1}
USE_SOURCE_SGL_KERNEL=${USE_SOURCE_SGL_KERNEL:-0}
COPY_INCREMENTAL_KERNEL=${COPY_INCREMENTAL_KERNEL:-0}

CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-6}
CSD_KEY_SELECTION_STRATEGY=${CSD_KEY_SELECTION_STRATEGY:-frequency}
CSD_SCORE_THRESHOLD=${CSD_SCORE_THRESHOLD:-0}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
CSD_TABLE_PROB_RATIO=${CSD_TABLE_PROB_RATIO:-0.3}
PLAIN_CSD_TABLE_PROB_RATIO=${PLAIN_CSD_TABLE_PROB_RATIO:-0.01}
CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO=${CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO:-1}
CSD_REBUILD_TOP_KEEP=${CSD_REBUILD_TOP_KEEP:-}
ENTROPY_P20_THRESHOLD=${ENTROPY_P20_THRESHOLD:-1.5638477802276611}
ENTROPY_P30_THRESHOLD=${ENTROPY_P30_THRESHOLD:-1.3415851593017578}
ENTROPY_P40_THRESHOLD=${ENTROPY_P40_THRESHOLD:-1.1563854217529297}

if [[ ! -f "${PLAIN_CSD_TABLE_PATH}" ]]; then
  echo "CSD table not found: ${PLAIN_CSD_TABLE_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT_BASE}/results" "${DRIVER_LOG_DIR}"

status=0
echo "CUDA_DEVICES=${CUDA_DEVICES} PORT=${PORT} TP_SIZE=${TP_SIZE} METHOD_SET=${METHOD_SET}"
echo "CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO=${CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO} CSD_REBUILD_TOP_KEEP=${CSD_REBUILD_TOP_KEEP:-<empty>}"

env \
  RUN_STAMP="${RUN_STAMP}" \
  OUT_ROOT="${OUT_ROOT_BASE}" \
  RESULT_FILE="${MERGED_RESULT_FILE}" \
  MODEL_PATH="${MODEL_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  TP_SIZE="${TP_SIZE}" \
  PORT="${PORT}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
  WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT}" \
  MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY}" \
  CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE}" \
  SHOW_KERNEL="${SHOW_KERNEL}" \
  USE_SOURCE_SGL_KERNEL="${USE_SOURCE_SGL_KERNEL}" \
  COPY_INCREMENTAL_KERNEL="${COPY_INCREMENTAL_KERNEL}" \
  TREE_SHAPES="${TREE_SHAPES}" \
  TASK_FILTER="${TASK_FILTER}" \
  METHOD_SET="${METHOD_SET}" \
  PLAIN_CSD_TABLE_PATH="${PLAIN_CSD_TABLE_PATH}" \
  CSD_FREQ_THRESHOLD="${CSD_FREQ_THRESHOLD}" \
  CSD_KEY_SELECTION_STRATEGY="${CSD_KEY_SELECTION_STRATEGY}" \
  CSD_SCORE_THRESHOLD="${CSD_SCORE_THRESHOLD}" \
  CSD_PROB_RATIO="${CSD_PROB_RATIO}" \
  CSD_TABLE_PROB_RATIO="${CSD_TABLE_PROB_RATIO}" \
  PLAIN_CSD_TABLE_PROB_RATIO="${PLAIN_CSD_TABLE_PROB_RATIO}" \
  CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO="${CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO}" \
  CSD_REBUILD_TOP_KEEP="${CSD_REBUILD_TOP_KEEP}" \
  ENTROPY_P20_THRESHOLD="${ENTROPY_P20_THRESHOLD}" \
  ENTROPY_P30_THRESHOLD="${ENTROPY_P30_THRESHOLD}" \
  ENTROPY_P40_THRESHOLD="${ENTROPY_P40_THRESHOLD}" \
  bash benchmark/csd/eval/run_lighteval_csd_classic_tree_shape_sweep.sh \
  >"${DRIVER_LOG_DIR}/rerun.log" 2>&1 || status=$?

python - "${MERGED_RESULT_FILE}" <<'PY_SUMMARY'
import json
import sys
from collections import Counter

path = sys.argv[1]
rows = []
try:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
except FileNotFoundError:
    pass

methods = []
for row in rows:
    tag = row.get("other", {}).get("run_tag", "")
    if tag.startswith("dynamic_entropy_p20_ignore_ratio_"):
        method = "dynamic_entropy_p20_ignore_ratio"
    elif tag.startswith("dynamic_ignore_ratio_"):
        method = "dynamic_ignore_ratio"
    elif tag.startswith("plain_entropy_p20_"):
        method = "plain_entropy_p20"
    else:
        method = tag.split("_", 1)[0] if tag else "unknown"
    methods.append(method)

print({"merged_result": path, "rows": len(rows), "methods": dict(Counter(methods))})
PY_SUMMARY

echo "Done. OUT_ROOT_BASE=${OUT_ROOT_BASE}"
echo "MERGED_RESULT_FILE=${MERGED_RESULT_FILE}"
exit "${status}"
