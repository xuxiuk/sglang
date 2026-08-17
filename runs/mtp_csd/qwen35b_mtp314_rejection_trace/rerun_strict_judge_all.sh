#!/usr/bin/env bash
# Re-judge all existing replay samples with the strict equivalence prompt.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/root/sglang-dspark-csd}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
INPUT_RUN=${INPUT_RUN:-${HERE}/runs/full_pipeline_20260813_015303}
OUTPUT_ROOT=${OUTPUT_ROOT:-${INPUT_RUN}/strict_rejudge_$(date +%Y%m%d_%H%M%S)}
TASKS=${TASKS:-"lcb_v6 aime25 olympiad_math_en"}

JUDGE_MODEL=${JUDGE_MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:30000/v1}
JUDGE_API_KEY=${JUDGE_API_KEY:-EMPTY}
JUDGE_PARALLEL_REQUESTS=${JUDGE_PARALLEL_REQUESTS:-64}
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-3600}
JUDGE_MAX_RETRIES=${JUDGE_MAX_RETRIES:-1}
JUDGE_GPU_SET=${JUDGE_GPU_SET:-0,1,2,3,4,5,6,7}
JUDGE_TP_SIZE=${JUDGE_TP_SIZE:-8}
JUDGE_DP_SIZE=${JUDGE_DP_SIZE:-8}
JUDGE_EP_SIZE=${JUDGE_EP_SIZE:-1}
JUDGE_MAX_RUNNING_REQUESTS=${JUDGE_MAX_RUNNING_REQUESTS:-64}
JUDGE_MEM_FRACTION_STATIC=${JUDGE_MEM_FRACTION_STATIC:-0.88}
JUDGE_CONTEXT_LENGTH=${JUDGE_CONTEXT_LENGTH:-96000}
JUDGE_PORT=${JUDGE_PORT:-30000}
JUDGE_DIST_INIT_ADDR=${JUDGE_DIST_INIT_ADDR:-127.0.0.1:30010}
JUDGE_NCCL_PORT=${JUDGE_NCCL_PORT:-30020}
JUDGE_READY_TIMEOUT=${JUDGE_READY_TIMEOUT:-1800}

test ! -e "${OUTPUT_ROOT}" || { echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"
for task in ${TASKS}; do
  test -s "${INPUT_RUN}/${task}/replay/replayed.jsonl"
done

judge_pid=
cleanup() {
  if [[ -n "${judge_pid}" ]]; then
    kill "${judge_pid}" 2>/dev/null || true
    wait "${judge_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES="${JUDGE_GPU_SET}"
export NCCL_IB_DISABLE=1
export NCCL_NET=${NCCL_NET:-Socket}
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

"${PYTHON}" -m sglang.launch_server \
  --model-path "${JUDGE_MODEL}" \
  --tp "${JUDGE_TP_SIZE}" --dp-size "${JUDGE_DP_SIZE}" --ep-size "${JUDGE_EP_SIZE}" \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune --swa-full-tokens-ratio 0.2 \
  --chunked-prefill-size 2048 --mem-fraction-static "${JUDGE_MEM_FRACTION_STATIC}" \
  --context-length "${JUDGE_CONTEXT_LENGTH}" \
  --cuda-graph-max-bs "${JUDGE_MAX_RUNNING_REQUESTS}" \
  --max-running-requests "${JUDGE_MAX_RUNNING_REQUESTS}" \
  --disable-radix-cache --trust-remote-code \
  --host 127.0.0.1 --port "${JUDGE_PORT}" \
  --dist-init-addr "${JUDGE_DIST_INIT_ADDR}" --nccl-port "${JUDGE_NCCL_PORT}" \
  >"${OUTPUT_ROOT}/judge_server.log" 2>&1 &
judge_pid=$!

deadline=$((SECONDS + JUDGE_READY_TIMEOUT))
until curl -fsS -H "Authorization: Bearer ${JUDGE_API_KEY}" \
    "${JUDGE_BASE_URL%/}/models" >/dev/null; do
  kill -0 "${judge_pid}" 2>/dev/null || {
    echo "Judge server exited; see ${OUTPUT_ROOT}/judge_server.log" >&2
    exit 1
  }
  (( SECONDS < deadline )) || { echo "Judge startup timed out" >&2; exit 1; }
  sleep 2
done

outputs=()
for task in ${TASKS}; do
  echo "[$(date -Is)] strict judge ${task} start"
  output="${OUTPUT_ROOT}/${task}/judge/judged_strict.jsonl"
  "${PYTHON}" "${HERE}/judge_replayed_branches.py" \
    --input "${INPUT_RUN}/${task}/replay/replayed.jsonl" \
    --output "${output}" \
    --base-url "${JUDGE_BASE_URL}" --model "${JUDGE_MODEL}" \
    --api-key "${JUDGE_API_KEY}" --parallel-requests "${JUDGE_PARALLEL_REQUESTS}" \
    --timeout "${JUDGE_TIMEOUT}" --max-retries "${JUDGE_MAX_RETRIES}" \
    --temperature 0 --sample-ratio 1 --seed 42
  outputs+=("${output}")
  echo "[$(date -Is)] strict judge ${task} complete"
done

"${PYTHON}" "${HERE}/summarize_judgments.py" \
  --inputs "${outputs[@]}" \
  --output-json "${OUTPUT_ROOT}/summary_strict.json" \
  --output-md "${OUTPUT_ROOT}/summary_strict.md"

echo "Strict re-judge complete: ${OUTPUT_ROOT}"
