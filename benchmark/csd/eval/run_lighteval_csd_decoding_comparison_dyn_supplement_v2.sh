#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

BASE_SCRIPT=${BASE_SCRIPT:-"${SCRIPT_DIR}/run_lighteval_csd_decoding_comparison_dyn.sh"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_decoding_comparison_supplement/${RUN_STAMP}"}
RESULT_FILE=${RESULT_FILE:-"${OUT_DIR}/results/csd_decoding_comparison_supplement.jsonl"}

# ==================== 模型路径控制 (新加) ====================
MODEL_PATH=${MODEL_PATH:-"/root/model/Qwen3.6-35B-A3B"}

# ==================== 硬件与网络控制 ====================
CUDA_DEVICES=${CUDA_DEVICES:-"4,5,6,7"}
TP_SIZE=${TP_SIZE:-4}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.9}
PORT=${PORT:-30002}

CSD_PROB_RATIOS=${CSD_PROB_RATIOS:-"0.3"}
CSD_REBUILD_TOP_KEEPS=${CSD_REBUILD_TOP_KEEPS:-"2000 5000 10000 15000 20000"}
SPEC_TOPK=${SPEC_TOPK:-1}
SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-5}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-5}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}

mkdir -p "${OUT_DIR}/results"

run_one() {
  local method="$1"
  local prob_ratio="$2"
  local top_keep="$3"
  local ignore_prob_ratio="$4"

  echo "------------------------------------------------------------"
  echo "Running supplemental sweep item:"
  echo "  METHOD                          : ${method}"
  echo "  MODEL_PATH                      : ${MODEL_PATH}"
  echo "  CSD_PROB_RATIO                  : ${prob_ratio}"
  echo "  CSD_REBUILD_TOP_KEEP            : ${top_keep:-None}"
  echo "  DYNAMIC_UPDATE_IGNORE_PROB_RATIO: ${ignore_prob_ratio}"
  echo "  CUDA_DEVICES                    : ${CUDA_DEVICES}"
  echo "  TP_SIZE                         : ${TP_SIZE}"
  echo "  PORT                            : ${PORT}"
  echo "------------------------------------------------------------"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  PORT="${PORT}" \
  TP_SIZE="${TP_SIZE}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  MODEL_PATH="${MODEL_PATH}" \
  TOKENIZER_PATH="${MODEL_PATH}" \
  RUN_STAMP="${RUN_STAMP}" \
  OUT_DIR="${OUT_DIR}" \
  RESULT_FILE="${RESULT_FILE}" \
  METHOD_FILTER="${method}" \
  CSD_PROB_RATIO="${prob_ratio}" \
  CSD_REBUILD_TOP_KEEP="${top_keep}" \
  CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO="${ignore_prob_ratio}" \
  SPEC_TOPK="${SPEC_TOPK}" \
  SPEC_NUM_STEPS="${SPEC_NUM_STEPS}" \
  SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS}" \
    bash "${BASE_SCRIPT}"
}

run_with_error_policy() {
  local method="$1"
  local prob_ratio="$2"
  local top_keep="$3"
  local ignore_prob_ratio="$4"
  local status=0

  run_one "${method}" "${prob_ratio}" "${top_keep}" "${ignore_prob_ratio}" || status=$?
  if [[ "${status}" != "0" ]]; then
    echo "Supplemental run failed: method=${method}, ratio=${prob_ratio}, top_keep=${top_keep:-none}, ignore_prob_ratio=${ignore_prob_ratio}, status=${status}" >&2
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit "${status}"
    fi
    return "${status}"
  fi
  return 0
}

status=0

# Missing ablation 1: pure dynamic update, no rebuild cap. Existing sweeps ran static and top_keep; this fills dynamic_update-only.
for prob_ratio in ${CSD_PROB_RATIOS}; do
  run_with_error_policy "csd_plain_table" "${prob_ratio}" "" "0" || status=$?
done

# Missing ablation 2: pure dynamic update with ratio-gate ignored during pair collection.
for prob_ratio in ${CSD_PROB_RATIOS}; do
  run_with_error_policy "csd_plain_table" "${prob_ratio}" "" "1" || status=$?
done

# Missing ablation 3: dynamic update with rebuild cap and ratio-gate ignored during pair collection.
for prob_ratio in ${CSD_PROB_RATIOS}; do
  for top_keep in ${CSD_REBUILD_TOP_KEEPS}; do
    run_with_error_policy "csd_plain_table_top5" "${prob_ratio}" "${top_keep}" "1" || status=$?
  done
done

echo "================================================------------"
echo "Supplemental CSD sweep done."
echo "OUT_DIR=${OUT_DIR}"
echo "RESULT_FILE=${RESULT_FILE}"
exit "${status}"
