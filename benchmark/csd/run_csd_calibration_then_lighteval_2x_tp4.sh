#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

CALIBRATION_SCRIPT=${CALIBRATION_SCRIPT:-"${REPO_ROOT}/benchmark/csd/redpajama/run_csd_calibration.sh"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"${REPO_ROOT}/benchmark/csd/eval/run_lighteval_csd_decoding_comparison_2x_tp4.sh"}

run_step() {
  local name="$1"
  local script="$2"

  echo "========== ${name}: start =========="
  bash "${script}"
  local status=$?
  echo "========== ${name}: exit ${status} =========="
  return "${status}"
}

calibration_status=0
run_step "redpajama calibration" "${CALIBRATION_SCRIPT}" || calibration_status=$?

eval_status=0
run_step "lighteval decoding comparison 2x tp4" "${EVAL_SCRIPT}" || eval_status=$?

echo "redpajama calibration status: ${calibration_status}"
echo "lighteval decoding comparison status: ${eval_status}"

if [[ "${calibration_status}" != "0" || "${eval_status}" != "0" ]]; then
  exit 1
fi
