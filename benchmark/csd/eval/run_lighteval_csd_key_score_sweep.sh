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
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_key_score_sweep/${RUN_STAMP}"}
RESULT_FILE=${RESULT_FILE:-"${OUT_DIR}/results/csd_key_score_sweep.jsonl"}

# Thresholds are calibrated on the gated ratio=0.3 RedPajama table. Scores have
# many ties, so threshold filtering retains approximately 4.60%, 5.05%, and
# 11.13% of that table rather than exactly 3%, 5%, and 10%.
CSD_SCORE_THRESHOLDS=${CSD_SCORE_THRESHOLDS:-"1.0 0.571428571429 0.333333333333"}
CSD_TARGET_RETAIN_PERCENTS=${CSD_TARGET_RETAIN_PERCENTS:-"3 5 10"}

# count_squared_over_total ignores the legacy frequency gate in CSDRuntime.
# Keep this at 1 for clear experiment metadata and backward-compatible configs.
CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-1}
CSD_KEY_SELECTION_STRATEGY=${CSD_KEY_SELECTION_STRATEGY:-count_squared_over_total}
METHOD=${METHOD:-csd_ratio_table_static}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}

mkdir -p "${OUT_DIR}/results"

status=0
read -r -a score_thresholds <<<"${CSD_SCORE_THRESHOLDS}"
read -r -a target_retain_percents <<<"${CSD_TARGET_RETAIN_PERCENTS}"
if [[ "${#score_thresholds[@]}" != "${#target_retain_percents[@]}" ]]; then
  echo "CSD_SCORE_THRESHOLDS and CSD_TARGET_RETAIN_PERCENTS must have the same number of values." >&2
  exit 2
fi

for index in "${!score_thresholds[@]}"; do
  score_threshold="${score_thresholds[index]}"
  target_retain_percent="${target_retain_percents[index]}"
  echo "Running ${METHOD}: target=${target_retain_percent}%, strategy=${CSD_KEY_SELECTION_STRATEGY}, score_threshold=${score_threshold}, freq_threshold=${CSD_FREQ_THRESHOLD}"
  run_status=0
  RUN_STAMP="${RUN_STAMP}" \
  OUT_DIR="${OUT_DIR}" \
  RESULT_FILE="${RESULT_FILE}" \
  METHOD_SET="${METHOD}" \
  METHOD_FILTER="${METHOD}" \
  CSD_KEY_SELECTION_STRATEGY="${CSD_KEY_SELECTION_STRATEGY}" \
  CSD_SCORE_THRESHOLD="${score_threshold}" \
  CSD_TARGET_RETAIN_PERCENT="${target_retain_percent}" \
  CSD_FREQ_THRESHOLD="${CSD_FREQ_THRESHOLD}" \
    bash "${BASE_SCRIPT}" || run_status=$?

  if [[ "${run_status}" != "0" ]]; then
    status="${run_status}"
    echo "Score sweep failed: target=${target_retain_percent}%, threshold=${score_threshold}, status=${run_status}" >&2
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit "${run_status}"
    fi
  fi
done

echo "Done."
echo "OUT_DIR=${OUT_DIR}"
echo "RESULT_FILE=${RESULT_FILE}"
exit "${status}"
