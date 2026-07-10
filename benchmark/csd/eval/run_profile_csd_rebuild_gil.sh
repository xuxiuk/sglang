#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

PY_SPY=${PY_SPY:-"${HOME}/.local/bin/py-spy"}
TABLE_PATH=${TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/runs/redpajama/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json"}
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmark/csd/runs/csd_rebuild_cpu_profile"}
ITERATIONS=${ITERATIONS:-200}
RATE=${RATE:-100}
NATIVE_EXTENSION=${NATIVE_EXTENSION:-}
STATEFUL_NATIVE=${STATEFUL_NATIVE:-0}

mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${NATIVE_EXTENSION}" ]]; then
  EXTRA_ARGS+=(--native-extension "${NATIVE_EXTENSION}")
fi
if [[ "${STATEFUL_NATIVE}" == "1" ]]; then
  EXTRA_ARGS+=(--stateful-native)
fi

"${PY_SPY}" record \
  --gil \
  --threads \
  --rate "${RATE}" \
  --format chrometrace \
  --output "${OUT_DIR}/csd_rebuild_gil.chrometrace.json" \
  -- python "${SCRIPT_DIR}/profile_csd_rebuild_cpu.py" \
  --table-path "${TABLE_PATH}" \
  --iterations "${ITERATIONS}" \
  "${EXTRA_ARGS[@]}" \
  | tee "${OUT_DIR}/summary.txt"

echo "Perfetto trace: ${OUT_DIR}/csd_rebuild_gil.chrometrace.json"
