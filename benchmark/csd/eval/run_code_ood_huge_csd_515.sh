#!/usr/bin/env bash
# Run APPS, TACO, and LCB-v6 code OOD generation with dynamic huge-threshold controls.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_code_ood_huge_csd_515}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/benchmark/csd/runs/domain_ood_csd_515/${RUN_STAMP}"}
DRIVER_LOG=${DRIVER_LOG:-"${OUT_ROOT}/driver.log"}

APPS_DATA=${APPS_DATA:-"${REPO_ROOT}/benchmark/csd/runs/code_ood_data/apps_2000_code_ood_alpaca.json"}
TACO_DATA=${TACO_DATA:-"${REPO_ROOT}/benchmark/csd/runs/code_ood_data/taco_1000_code_ood_alpaca.json"}
LCB_DATA=${LCB_DATA:-"${REPO_ROOT}/benchmark/csd/runs/code_ood_data/lcb_v6_175_code_ood_alpaca.json"}

mkdir -p "${OUT_ROOT}"

for data_file in "${APPS_DATA}" "${TACO_DATA}" "${LCB_DATA}"; do
  if [[ ! -f "${data_file}" ]]; then
    echo "Missing code OOD data file: ${data_file}" >&2
    echo "Run: python benchmark/csd/eval/prepare_code_ood.py --datasets apps,taco,lcb_v6 --limit 2000" >&2
    exit 1
  fi
done

cat <<EOF | tee "${OUT_ROOT}/launcher_config.txt"
OUT_ROOT=${OUT_ROOT}
DRIVER_LOG=${DRIVER_LOG}
APPS_DATA=${APPS_DATA}
TACO_DATA=${TACO_DATA}
LCB_DATA=${LCB_DATA}
CUDA_DEVICES=${CUDA_DEVICES:-4,5,6,7}
PORT=${PORT:-30072}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-81920}
CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD:-4096}
CSD_HUGE_REBUILD_THRESHOLD=${CSD_HUGE_REBUILD_THRESHOLD:-1000000000}
METHOD_SET=${METHOD_SET:-plain dynamic_ignore_ratio dynamic_ignore_ratio_huge dynamic_entropy_p20_ignore_ratio dynamic_entropy_p20_ignore_ratio_huge}
EOF

env \
  RUN_STAMP="${RUN_STAMP}" \
  OUT_ROOT="${OUT_ROOT}" \
  CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" \
  PORT="${PORT:-30072}" \
  OOD_DATA_FILES="${APPS_DATA} ${TACO_DATA} ${LCB_DATA}" \
  METHOD_SET="${METHOD_SET:-plain dynamic_ignore_ratio dynamic_ignore_ratio_huge dynamic_entropy_p20_ignore_ratio dynamic_entropy_p20_ignore_ratio_huge}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-81920}" \
  ENABLE_THINKING="${ENABLE_THINKING:-true}" \
  CSD_REBUILD_THRESHOLD="${CSD_REBUILD_THRESHOLD:-4096}" \
  CSD_HUGE_REBUILD_THRESHOLD="${CSD_HUGE_REBUILD_THRESHOLD:-1000000000}" \
  PARALLEL="${PARALLEL:-24}" \
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-24}" \
  ENABLE_SERVER_METRICS="${ENABLE_SERVER_METRICS:-true}" \
  METRICS_SCRAPE_INTERVAL="${METRICS_SCRAPE_INTERVAL:-30}" \
  bash benchmark/csd/eval/run_domain_ood_csd_515.sh \
  >"${DRIVER_LOG}" 2>&1

echo "Done."
echo "OUT_ROOT=${OUT_ROOT}"
echo "DRIVER_LOG=${DRIVER_LOG}"
