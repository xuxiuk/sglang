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
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_above_uniform_share/${RUN_STAMP}"}
RESULT_FILE=${RESULT_FILE:-"${OUT_DIR}/results/csd_above_uniform_share.jsonl"}

# Run one same-round EAGLE baseline and CSD table comparison runs built by:
#   count(d, t) / count(d, *) > 1 / K(d)
# where K(d) is the number of distinct replacements observed for draft token d.
#
# Sweep layout:
#   - vanilla EAGLE baseline once
#   - for each min_count in CSD_MIN_COUNTS:
#       1. static table
#       2. dynamic update, respecting the prob-ratio gate for recorded pairs
#       3. dynamic update, ignoring the prob-ratio gate for recorded pairs
#
# Dynamic runs intentionally use csd_ratio_table, not csd_ratio_table_top5, so
# no dynamic top_keep is applied.
CSD_MIN_COUNTS=${CSD_MIN_COUNTS:-"2 3 4 5 6"}
CSD_KEY_SELECTION_STRATEGY=${CSD_KEY_SELECTION_STRATEGY:-above_uniform_share}
CSD_SCORE_THRESHOLD=${CSD_SCORE_THRESHOLD:-0}
RUN_EAGLE_BASELINE=${RUN_EAGLE_BASELINE:-1}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}

mkdir -p "${OUT_DIR}/results"

run_one() {
  local method_set="$1"
  local min_count="$2"
  local ignore_prob_ratio="$3"
  local label="$4"
  local status=0

  echo "------------------------------------------------------------"
  echo "Running above-uniform-share run item:"
  echo "  METHOD_SET                         : ${method_set}"
  echo "  CSD_KEY_SELECTION_STRATEGY          : ${CSD_KEY_SELECTION_STRATEGY}"
  echo "  CSD_FREQ_THRESHOLD / min_count      : ${min_count}"
  echo "  CSD_SCORE_THRESHOLD                 : ${CSD_SCORE_THRESHOLD}"
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
  CSD_FREQ_THRESHOLD="${min_count}" \
  CSD_SCORE_THRESHOLD="${CSD_SCORE_THRESHOLD}" \
  CSD_TARGET_RETAIN_PERCENT="${label}" \
  CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO="${ignore_prob_ratio}" \
    bash "${BASE_SCRIPT}" || status=$?

  if [[ "${status}" != "0" ]]; then
    echo "Run item failed: method_set=${method_set}, strategy=${CSD_KEY_SELECTION_STRATEGY}, min_count=${min_count}, ignore=${ignore_prob_ratio}, status=${status}" >&2
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit "${status}"
    fi
  fi
}

if [[ "${RUN_EAGLE_BASELINE}" == "1" ]]; then
  # EAGLE baseline means speculative vanilla without CSD.
  run_one "vanilla" 1 0 "eagle_baseline"
fi

for min_count in ${CSD_MIN_COUNTS}; do
  run_one "csd_ratio_table_static" "${min_count}" 0 "mincount${min_count}_static"
  run_one "csd_ratio_table" "${min_count}" 0 "mincount${min_count}_dynamic_respectratio"
  run_one "csd_ratio_table" "${min_count}" 1 "mincount${min_count}_dynamic_ignoreratio"
done

echo "Done."
echo "OUT_DIR=${OUT_DIR}"
echo "RESULT_FILE=${RESULT_FILE}"
