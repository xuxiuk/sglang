#!/usr/bin/env bash
# Canonical v0.5.10 five-method MTP 5/1/5 + CSD evaluation on six datasets.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_mtp515_v0510_five_methods_mr48}
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/benchmark/csd/runs/${RUN_STAMP}}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang/bin/python}
CUDA_DEVICES=${CUDA_DEVICES:-4,5,6,7}
PORT=${PORT:-30320}
DEFAULT_TABLE=${REPO_ROOT}/benchmark/csd/assets/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json
CALIBRATION_TABLE_PATH=${CALIBRATION_TABLE_PATH:-${RUN_ROOT}/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json}
TABLE=${TABLE:-${DEFAULT_TABLE}}
GEN_KWARGS=${GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0,seed=1234}
METHODS=${METHODS:-"auto eagle plain dynamic dynamic_entropy_p20_ignore_ratio"}
OOD_DATA_FILES=${OOD_DATA_FILES:-"${REPO_ROOT}/benchmark/csd/assets/data/apps_2000_code_ood_alpaca.json ${REPO_ROOT}/benchmark/csd/assets/data/taco_1000_code_ood_alpaca.json"}
RUN_CALIBRATION=${RUN_CALIBRATION:-0}
CALIBRATION_ONLY=${CALIBRATION_ONLY:-0}

if [[ "${CALIBRATION_ONLY}" == "1" && "${RUN_CALIBRATION}" != "1" ]]; then
  echo "CALIBRATION_ONLY=1 requires RUN_CALIBRATION=1." >&2
  exit 1
fi

if [[ "${RUN_CALIBRATION}" == "1" ]]; then
  echo "[0/2] Build the matching RedPajama calibration table"
  TABLE="${CALIBRATION_TABLE_PATH}"
  env \
    PYTHON="${PYTHON}" \
    MODEL_PATH="${MODEL_PATH:-/root/model/Qwen3.5-35B-A3B}" \
    CUDA_DEVICES="${CUDA_DEVICES}" \
    PORT="${PORT}" \
    OUT_DIR="${RUN_ROOT}/calibration" \
    CSD_TABLE_PATH="${CALIBRATION_TABLE_PATH}" \
    bash benchmark/csd/calibration/run_redpajama_calibration_515.sh
  echo "Use freshly calibrated table for all CSD evaluation methods: ${TABLE}"
fi

if [[ ! -s "${TABLE}" ]]; then
  echo "Missing MTP calibration table: ${TABLE}" >&2
  exit 1
fi
if [[ "${CALIBRATION_ONLY}" == "1" ]]; then
  echo "Calibration complete: ${TABLE}"
  exit 0
fi
for data_file in ${OOD_DATA_FILES}; do
  if [[ ! -s "${data_file}" ]]; then
    echo "Missing code dataset: ${data_file}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}/driver_logs"
cat >"${RUN_ROOT}/version_compare_config.txt" <<EOF
started_at=$(date -Is)
repo=${REPO_ROOT}
git_head=$(git rev-parse HEAD)
git_describe=$(git describe --tags --always)
python=${PYTHON}
cuda_devices=${CUDA_DEVICES}
methods=${METHODS}
tree=515:5:1:5
max_running_requests=48
generation=${GEN_KWARGS}
lcb_aime_math_max_new_tokens=81920
gsm8k_max_new_tokens=32768
apps_taco_max_new_tokens=40960
deterministic_inference=disabled
calibration_requested=${RUN_CALIBRATION}
table=${TABLE}
table_sha256=$(sha256sum "${TABLE}" | awk '{print $1}')
EOF
"${PYTHON}" - <<'PY' >>"${RUN_ROOT}/version_compare_config.txt"
import torch, sglang, sgl_kernel
print(f"torch={torch.__version__}")
print(f"sglang={sglang.__version__}")
print(f"sglang_path={sglang.__file__}")
print(f"kernel_path={sgl_kernel.__file__}")
PY

echo "[1/2] v0.5.10 LightEval five-method matrix" | tee "${RUN_ROOT}/driver_logs/supervisor.log"
env \
  PATH="$(dirname "${PYTHON}"):${PATH}" \
  LIGHTEVAL_PYTHON="${PYTHON}" \
  RUN_STAMP="${RUN_STAMP}" \
  OUT_ROOT="${RUN_ROOT}/lighteval" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  PORT="${PORT}" \
  TREE_SHAPES="515:5:1:5" \
  METHOD_SET="${METHODS}" \
  TASK_FILTER="lcb aime25 math500 gsm8k" \
  MAX_RUNNING_REQUESTS=48 \
  MAMBA_SCHEDULER_STRATEGY=no_buffer \
  PLAIN_CSD_TABLE_PATH="${TABLE}" \
  CSD_REBUILD_THRESHOLD=4096 \
  CODING_RECOMMENDED_GEN_KWARGS="${GEN_KWARGS}" \
  THINK_GENERAL_RECOMMENDED_GEN_KWARGS="${GEN_KWARGS}" \
  LCB_RECOMMENDED_GEN_KWARGS="${GEN_KWARGS}" \
  LCB_MAX_GEN_TOKS=81920 LCB_MAX_LENGTH=96000 \
  AIME_MAX_GEN_TOKS=81920 AIME_MAX_LENGTH=96000 \
  MATH500_MAX_GEN_TOKS=81920 MATH500_MAX_LENGTH=96000 \
  GSM8K_MAX_GEN_TOKS=32768 GSM8K_MAX_LENGTH=40960 \
  bash benchmark/csd/eval/libexec/lighteval_matrix.bash \
  2>&1 | tee "${RUN_ROOT}/driver_logs/lighteval.log"

echo "[2/2] v0.5.10 APPS/TACO five-method matrix" | tee -a "${RUN_ROOT}/driver_logs/supervisor.log"
env \
  PYTHON="${PYTHON}" \
  RUN_STAMP="${RUN_STAMP}" \
  OUT_ROOT="${RUN_ROOT}/code_ood" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  PORT="${PORT}" \
  METHOD_SET="${METHODS}" \
  OOD_DATA_FILES="${OOD_DATA_FILES}" \
  NUM_EXAMPLES=1000 \
  MAX_RUNNING_REQUESTS=48 \
  PARALLEL=24 \
  MAMBA_SCHEDULER_STRATEGY=no_buffer \
  MAX_NEW_TOKENS=40960 \
  TEMPERATURE=1.0 TOP_P=0.95 TOP_K=20 PRESENCE_PENALTY=1.5 \
  PLAIN_CSD_TABLE_PATH="${TABLE}" \
  CSD_REBUILD_THRESHOLD=4096 \
  bash benchmark/csd/eval/libexec/code_ood_matrix.bash \
  2>&1 | tee "${RUN_ROOT}/driver_logs/code.log"

echo "completed_at=$(date -Is)" | tee -a "${RUN_ROOT}/driver_logs/supervisor.log"
