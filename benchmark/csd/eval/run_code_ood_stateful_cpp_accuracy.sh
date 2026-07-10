#!/usr/bin/env bash
# Run coding OOD generation with saved outputs for downstream accuracy evaluation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_code_ood_stateful_cpp_accuracy}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/benchmark/csd/runs/${RUN_STAMP}"}
NATIVE_EXTENSION=${NATIVE_EXTENSION:-"${REPO_ROOT}/benchmark/csd/native/csd_rebuild_cpu_ext.so"}
METHOD_SET=${METHOD_SET:-"eagle plain dynamic_ignore_ratio dynamic_entropy_p20_ignore_ratio"}

APPS_DATA=${APPS_DATA:-"${REPO_ROOT}/benchmark/csd/runs/code_ood_data/apps_2000_code_ood_alpaca.json"}
TACO_DATA=${TACO_DATA:-"${REPO_ROOT}/benchmark/csd/runs/code_ood_data/taco_1000_code_ood_alpaca.json"}
LCB_DATA=${LCB_DATA:-"${REPO_ROOT}/benchmark/csd/runs/code_ood_data/lcb_v6_175_code_ood_alpaca.json"}

APPS_NUM_EXAMPLES=${APPS_NUM_EXAMPLES:-1000}
TACO_NUM_EXAMPLES=${TACO_NUM_EXAMPLES:-1000}
LCB_NUM_EXAMPLES=${LCB_NUM_EXAMPLES:-175}
CODE_MAX_NEW_TOKENS=${CODE_MAX_NEW_TOKENS:-40960}
LCB_MAX_NEW_TOKENS=${LCB_MAX_NEW_TOKENS:-81920}
CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD:-4096}
PARALLEL=${PARALLEL:-24}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-24}

APPS_CUDA_DEVICES=${APPS_CUDA_DEVICES:-0,1,2,3}
OTHER_CUDA_DEVICES=${OTHER_CUDA_DEVICES:-4,5,6,7}
APPS_PORT=${APPS_PORT:-30401}
OTHER_PORT=${OTHER_PORT:-30411}

if [[ -n "${NATIVE_EXTENSION}" && ! -f "${NATIVE_EXTENSION}" ]]; then
  echo "Standalone CSD extension not found; using the installed sgl-kernel op." >&2
  NATIVE_EXTENSION=""
fi
for data_file in "${APPS_DATA}" "${TACO_DATA}" "${LCB_DATA}"; do
  if [[ ! -f "${data_file}" ]]; then
    echo "Missing dataset: ${data_file}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}"

cat >"${OUT_ROOT}/launcher_config.txt" <<EOF
RUN_STAMP=${RUN_STAMP}
OUT_ROOT=${OUT_ROOT}
NATIVE_EXTENSION=${NATIVE_EXTENSION}
METHOD_SET=${METHOD_SET}
APPS_DATA=${APPS_DATA}
APPS_NUM_EXAMPLES=${APPS_NUM_EXAMPLES}
TACO_DATA=${TACO_DATA}
TACO_NUM_EXAMPLES=${TACO_NUM_EXAMPLES}
LCB_DATA=${LCB_DATA}
LCB_NUM_EXAMPLES=${LCB_NUM_EXAMPLES}
CODE_MAX_NEW_TOKENS=${CODE_MAX_NEW_TOKENS}
LCB_MAX_NEW_TOKENS=${LCB_MAX_NEW_TOKENS}
CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD}
PARALLEL=${PARALLEL}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}
APPS_CUDA_DEVICES=${APPS_CUDA_DEVICES}
OTHER_CUDA_DEVICES=${OTHER_CUDA_DEVICES}
EOF

run_dataset() {
  local label=$1
  local data_file=$2
  local num_examples=$3
  local max_new_tokens=$4
  local cuda_devices=$5
  local port=$6
  local out_dir="${OUT_ROOT}/${label}"

  mkdir -p "${out_dir}"
  local native_env=()
  if [[ -n "${NATIVE_EXTENSION}" ]]; then
    native_env+=(SGLANG_CSD_NATIVE_EXTENSION_PATH="${NATIVE_EXTENSION}")
  fi

  env "${native_env[@]}" \
    RUN_STAMP="${RUN_STAMP}_${label}" \
    OUT_ROOT="${out_dir}" \
    CUDA_DEVICES="${cuda_devices}" \
    PORT="${port}" \
    OOD_DATA_FILES="${data_file}" \
    NUM_EXAMPLES="${num_examples}" \
    METHOD_SET="${METHOD_SET}" \
    MAX_NEW_TOKENS="${max_new_tokens}" \
    CSD_REBUILD_THRESHOLD="${CSD_REBUILD_THRESHOLD}" \
    PARALLEL="${PARALLEL}" \
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
    ENABLE_SERVER_METRICS=true \
    METRICS_SCRAPE_INTERVAL=30 \
    bash benchmark/csd/eval/run_domain_ood_csd_515.sh \
    >"${out_dir}/driver.log" 2>&1
}

# APPS runs independently on GPU 0-3. TACO and LCB run sequentially on GPU 4-7.
run_dataset \
  apps "${APPS_DATA}" "${APPS_NUM_EXAMPLES}" "${CODE_MAX_NEW_TOKENS}" \
  "${APPS_CUDA_DEVICES}" "${APPS_PORT}" &
apps_pid=$!

(
  run_dataset \
    taco "${TACO_DATA}" "${TACO_NUM_EXAMPLES}" "${CODE_MAX_NEW_TOKENS}" \
    "${OTHER_CUDA_DEVICES}" "${OTHER_PORT}"
  run_dataset \
    lcb "${LCB_DATA}" "${LCB_NUM_EXAMPLES}" "${LCB_MAX_NEW_TOKENS}" \
    "${OTHER_CUDA_DEVICES}" "${OTHER_PORT}"
) &
other_pid=$!

cleanup() {
  kill "${apps_pid}" "${other_pid}" >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

status=0
wait "${apps_pid}" || status=$?
wait "${other_pid}" || status=$?
trap - INT TERM EXIT

if [[ "${status}" != "0" ]]; then
  echo "At least one coding benchmark group failed; inspect ${OUT_ROOT}/*/driver.log" >&2
  exit "${status}"
fi

echo "Done: ${OUT_ROOT}"
