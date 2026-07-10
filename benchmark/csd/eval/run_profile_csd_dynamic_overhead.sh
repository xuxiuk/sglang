#!/usr/bin/env bash
# Profile a small request batch for CSD dynamic/entropy overhead. This does not run LightEval.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="/root/miniconda3/envs/sglang/bin:${CUDA_HOME}/bin:/home/ccuser/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export SGLANG_SRC=${SGLANG_SRC:-"${REPO_ROOT}/python"}
export PYTHONPATH="${SGLANG_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"
# Keep trace size manageable.
export SGLANG_PROFILE_WITH_STACK=false
export SGLANG_PROFILE_RECORD_SHAPES=false
# Add torch profiler record_function ranges around CSD verify/update logic.
export SGLANG_CSD_PROFILE_ANNOTATIONS=${SGLANG_CSD_PROFILE_ANNOTATIONS:-1}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_dynamic_overhead_one_request}
BASE_OUT=${BASE_OUT:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_515_calib1024_entropy_ratio_sweep/20260629_515_calib1024_entropy_ratio_sweep/profiles"}
OUT_DIR=${OUT_DIR:-"${BASE_OUT}/${RUN_STAMP}"}

MODEL_PATH=${MODEL_PATH:-/root/model/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
CSD_TABLE_PATH=${CSD_TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/runs/redpajama/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json"}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
TP_SIZE=${TP_SIZE:-4}
PORT=${PORT:-30111}
METHODS=${METHODS:-"plain dynamic p20_no_ratio"}
WARMUP_TOKENS=${WARMUP_TOKENS:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
NUM_PROFILE_REQUESTS=${NUM_PROFILE_REQUESTS:-4}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-1}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.72}
MAX_LENGTH=${MAX_LENGTH:-8192}
CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD:-4096}

mkdir -p "${OUT_DIR}"

echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_DEVICES=${CUDA_DEVICES} TP_SIZE=${TP_SIZE} METHODS=${METHODS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS} WARMUP_TOKENS=${WARMUP_TOKENS} NUM_PROFILE_REQUESTS=${NUM_PROFILE_REQUESTS}"
echo "CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
python benchmark/csd/eval/profile_csd_dynamic_overhead.py \
  --model "${MODEL_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --csd-table-path "${CSD_TABLE_PATH}" \
  --csd-rebuild-threshold "${CSD_REBUILD_THRESHOLD}" \
  --out-dir "${OUT_DIR}" \
  --tp-size "${TP_SIZE}" \
  --port "${PORT}" \
  --methods ${METHODS} \
  --warmup-tokens "${WARMUP_TOKENS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-profile-requests "${NUM_PROFILE_REQUESTS}" \
  --num-warmup-requests "${NUM_WARMUP_REQUESTS}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --max-length "${MAX_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  "$@"
