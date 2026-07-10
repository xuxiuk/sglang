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
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_above_uniform_share_min6_prob_ratio_supplement/${RUN_STAMP}"}
RESULT_FILE=${RESULT_FILE:-"${OUT_DIR}/results/csd_above_uniform_share_min6_prob_ratio_supplement.jsonl"}

CSD_MIN_COUNT=${CSD_MIN_COUNT:-6}
CSD_PROB_RATIOS=${CSD_PROB_RATIOS:-"0.01 0.05 0.1 0.3 0.5"}
CSD_KEY_SELECTION_STRATEGY=${CSD_KEY_SELECTION_STRATEGY:-above_uniform_share}
CSD_SCORE_THRESHOLD=${CSD_SCORE_THRESHOLD:-0}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}

mkdir -p "${OUT_DIR}/results"

run_one() {
  local method_set="$1"
  local ignore_prob_ratio="$2"
  local prob_ratio="$3"
  local label="$4"
  local status=0

  echo "------------------------------------------------------------"
  echo "Running above-uniform-share min6 prob-ratio supplement item:"
  echo "  METHOD_SET                         : ${method_set}"
  echo "  CSD_KEY_SELECTION_STRATEGY          : ${CSD_KEY_SELECTION_STRATEGY}"
  echo "  CSD_FREQ_THRESHOLD / min_count      : ${CSD_MIN_COUNT}"
  echo "  CSD_SCORE_THRESHOLD                 : ${CSD_SCORE_THRESHOLD}"
  echo "  CSD_PROB_RATIO                      : ${prob_ratio}"
  echo "  CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO: ${ignore_prob_ratio}"
  echo "  LABEL                              : ${label}"
  echo "  OUT_DIR                            : ${OUT_DIR}"
  echo "  RESULT_FILE                        : ${RESULT_FILE}"
  echo "------------------------------------------------------------"

  RUN_STAMP="${RUN_STAMP}" \
  OUT_DIR="${OUT_DIR}" \
  RESULT_FILE="${RESULT_FILE}" \
  METHOD_SET="${method_set}" \
  METHOD_FILTER="all" \
  CSD_KEY_SELECTION_STRATEGY="${CSD_KEY_SELECTION_STRATEGY}" \
  CSD_FREQ_THRESHOLD="${CSD_MIN_COUNT}" \
  CSD_SCORE_THRESHOLD="${CSD_SCORE_THRESHOLD}" \
  CSD_PROB_RATIO="${prob_ratio}" \
  CSD_TARGET_RETAIN_PERCENT="${label}" \
  CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO="${ignore_prob_ratio}" \
    bash "${BASE_SCRIPT}" || status=$?

  if [[ "${status}" != "0" ]]; then
    echo "Run item failed: method_set=${method_set}, strategy=${CSD_KEY_SELECTION_STRATEGY}, min_count=${CSD_MIN_COUNT}, prob_ratio=${prob_ratio}, ignore=${ignore_prob_ratio}, status=${status}" >&2
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit "${status}"
    fi
  fi
}

for prob_ratio in ${CSD_PROB_RATIOS}; do
  run_one "csd_ratio_table" 0 "${prob_ratio}" "mincount${CSD_MIN_COUNT}_ratio${prob_ratio}_dynamic_respectratio"
done

echo "Done."
echo "OUT_DIR=${OUT_DIR}"
echo "RESULT_FILE=${RESULT_FILE}"
