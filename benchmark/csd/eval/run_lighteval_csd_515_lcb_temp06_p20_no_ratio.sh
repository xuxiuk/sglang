#!/usr/bin/env bash
# LCB-only rerun for Qwen3.5-35B-A3B using the model-card precise coding
# sampling parameters while keeping the 515 CSD table and max_new_tokens=81920.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_515_lcb_temp06_p20_no_ratio}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_515_lcb_temp06_p20_no_ratio/${RUN_STAMP}"}
RESULT_FILE=${RESULT_FILE:-"${OUT_ROOT}/results/lcb_temp06_p20_no_ratio.jsonl"}
DRIVER_LOG=${DRIVER_LOG:-"${OUT_ROOT}/driver.log"}

PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/runs/redpajama/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json"}

mkdir -p "${OUT_ROOT}/results" "$(dirname "${DRIVER_LOG}")"

if [[ ! -f "${PLAIN_CSD_TABLE_PATH}" ]]; then
  echo "CSD table not found: ${PLAIN_CSD_TABLE_PATH}" >&2
  exit 1
fi

echo "OUT_ROOT=${OUT_ROOT}"
echo "RESULT_FILE=${RESULT_FILE}"
echo "DRIVER_LOG=${DRIVER_LOG}"
echo "PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH}"

env \
  RUN_STAMP="${RUN_STAMP}" \
  OUT_ROOT="${OUT_ROOT}" \
  RESULT_FILE="${RESULT_FILE}" \
  CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" \
  PORT="${PORT:-30016}" \
  TP_SIZE="${TP_SIZE:-4}" \
  TASK_FILTER="${TASK_FILTER:-lcb}" \
  TREE_SHAPES="${TREE_SHAPES:-515:5:1:5}" \
  METHOD_SET="${METHOD_SET:-plain dynamic dynamic_ignore_ratio plain_entropy_p20 dynamic_entropy_p20_ignore_ratio}" \
  PLAIN_CSD_TABLE_PATH="${PLAIN_CSD_TABLE_PATH}" \
  CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO="${CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO:-1}" \
  LCB_RECOMMENDED_GEN_KWARGS="${LCB_RECOMMENDED_GEN_KWARGS:-temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0}" \
  LCB_MAX_GEN_TOKS="${LCB_MAX_GEN_TOKS:-81920}" \
  LCB_MAX_LENGTH="${LCB_MAX_LENGTH:-96000}" \
  bash benchmark/csd/eval/run_lighteval_csd_classic_tree_shape_sweep.sh \
  >"${DRIVER_LOG}" 2>&1

echo "Done. RESULT_FILE=${RESULT_FILE}"
