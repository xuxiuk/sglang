#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
INPUT_RUN=${INPUT_RUN:-${HERE}/runs/full_pipeline_20260813_015303}
OUTPUT_ROOT=${OUTPUT_ROOT:-${INPUT_RUN}/v3_judge_$(date +%Y%m%d_%H%M%S)}
TASKS=${TASKS:-"lcb_v6 aime25 olympiad_math_en"}
MODEL=${MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
PORT=${PORT:-30000}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}/v1}
JUDGE_SCRIPT=${JUDGE_SCRIPT:-${HERE}/judge_replayed_branches_v3.py}
PROMPT_NAME=${PROMPT_NAME:-V3}
OUTPUT_BASENAME=${OUTPUT_BASENAME:-judged_v3.jsonl}
SUMMARY_JSON=${SUMMARY_JSON:-summary_v3.json}
SUMMARY_MD=${SUMMARY_MD:-summary_v3.md}
MAX_RETRIES=${MAX_RETRIES:-1}

test ! -e "${OUTPUT_ROOT}" || { echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"
for task in ${TASKS}; do test -s "${INPUT_RUN}/${task}/replay/replayed.jsonl"; done

server_pid=
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_IB_DISABLE=1
export NCCL_NET=${NCCL_NET:-Socket}
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

"${PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp 8 --dp-size 8 --ep-size 1 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune --swa-full-tokens-ratio 0.2 \
  --chunked-prefill-size 2048 --mem-fraction-static 0.88 \
  --context-length 96000 --cuda-graph-max-bs 64 --max-running-requests 64 \
  --disable-radix-cache --trust-remote-code \
  --host 127.0.0.1 --port "${PORT}" \
  --dist-init-addr 127.0.0.1:30010 --nccl-port 30020 \
  >"${OUTPUT_ROOT}/judge_server.log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + 1800))
until curl -fsS "${BASE_URL%/}/models" >/dev/null; do
  kill -0 "${server_pid}" 2>/dev/null || { echo "V3 judge server exited" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "V3 judge startup timed out" >&2; exit 1; }
  sleep 2
done

outputs=()
for task in ${TASKS}; do
  echo "[$(date -Is)] ${PROMPT_NAME} judge ${task} start"
  output="${OUTPUT_ROOT}/${task}/judge/${OUTPUT_BASENAME}"
  "${PYTHON}" "${JUDGE_SCRIPT}" \
    --input "${INPUT_RUN}/${task}/replay/replayed.jsonl" \
    --output "${output}" --base-url "${BASE_URL}" --model "${MODEL}" \
    --parallel-requests 64 --timeout 3600 --max-retries "${MAX_RETRIES}" \
    --temperature 0 --max-tokens 1536 --seed 42 --resume
  outputs+=("${output}")
  echo "[$(date -Is)] ${PROMPT_NAME} judge ${task} complete"
done

"${PYTHON}" "${HERE}/summarize_judgments_v3.py" \
  --inputs "${outputs[@]}" \
  --output-json "${OUTPUT_ROOT}/${SUMMARY_JSON}" \
  --output-md "${OUTPUT_ROOT}/${SUMMARY_MD}"

echo "${PROMPT_NAME} judge complete: ${OUTPUT_ROOT}"
