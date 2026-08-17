#!/usr/bin/env bash
# Globally sample GSM8K rejection events. Replay the same sample in two ways:
# 128 tokens for the V4 judge, and to EOS for objective GSM8K gold scoring.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/root/sglang-dspark-csd}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
TRACE_ROOT=${TRACE_ROOT:-${HERE}/runs/gsm8k_counterfactual_20260814_191507/trace_run/trace}
OUTPUT_ROOT=${OUTPUT_ROOT:-${HERE}/runs/gsm8k_full_counterfactual_judge_$(date +%Y%m%d_%H%M%S)}

REPLAY_MODEL=${REPLAY_MODEL:-/data/model/Qwen3.5-35B-A3B}
REPLAY_CUDA_DEVICES=${REPLAY_CUDA_DEVICES:-0,1,2,3}
REPLAY_TP_SIZE=${REPLAY_TP_SIZE:-4}
REPLAY_BATCH_SIZE=${REPLAY_BATCH_SIZE:-32}
REPLAY_MAX_RUNNING_REQUESTS=${REPLAY_MAX_RUNNING_REQUESTS:-64}
REPLAY_MAX_NEW_TOKENS=${REPLAY_MAX_NEW_TOKENS:-8192}
REPLAY_TEMPERATURE=${REPLAY_TEMPERATURE:-0.0}
JUDGE_REPLAY_MAX_NEW_TOKENS=${JUDGE_REPLAY_MAX_NEW_TOKENS:-128}
EVENT_FILTER=${EVENT_FILTER:-all}
SAMPLE_SEED=${SAMPLE_SEED:-42}
SAMPLE_RATIO=${SAMPLE_RATIO:-0.03}

JUDGE_MODEL=${JUDGE_MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
JUDGE_PORT=${JUDGE_PORT:-30100}
JUDGE_BASE_URL=http://127.0.0.1:${JUDGE_PORT}/v1
JUDGE_PARALLEL_REQUESTS=${JUDGE_PARALLEL_REQUESTS:-64}

test -s "${TRACE_ROOT}/events.rank0.jsonl"
test -s "${TRACE_ROOT}/requests.rank0.jsonl"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${HERE}:${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

cat >"${OUTPUT_ROOT}/config.env" <<EOF
TRACE_ROOT=${TRACE_ROOT}
SELECTION=uniform global random sample from all matched rejection events
EVENT_FILTER=${EVENT_FILTER}
GLOBAL_EVENT_CAP=none
SAMPLE_SEED=${SAMPLE_SEED}
SAMPLE_RATIO=${SAMPLE_RATIO}
REPLAY_MODEL=${REPLAY_MODEL}
REPLAY_TEMPERATURE=${REPLAY_TEMPERATURE}
REPLAY_MAX_NEW_TOKENS=${REPLAY_MAX_NEW_TOKENS}
JUDGE_REPLAY_MAX_NEW_TOKENS=${JUDGE_REPLAY_MAX_NEW_TOKENS}
GOLD_SCORER=GSM8K final numeric answer
JUDGE_MODEL=${JUDGE_MODEL}
JUDGE_PROMPT=V4_BOUNDARY_CONSTRAINED_INDEPENDENT_VALIDITY
EOF

echo "[$(date -Is)] full-to-EOS replay for GSM8K gold scoring start"
"${PYTHON}" "${HERE}/replay_gsm8k_counterfactual.py" \
  --trace-dir "${TRACE_ROOT}" \
  --output "${OUTPUT_ROOT}/counterfactual_events.jsonl" \
  --summary "${OUTPUT_ROOT}/counterfactual_summary.json" \
  --model "${REPLAY_MODEL}" --tp-size "${REPLAY_TP_SIZE}" \
  --cuda-devices "${REPLAY_CUDA_DEVICES}" --mem-fraction-static 0.75 \
  --context-length 96000 --max-new-tokens "${REPLAY_MAX_NEW_TOKENS}" \
  --batch-size "${REPLAY_BATCH_SIZE}" \
  --max-running-requests "${REPLAY_MAX_RUNNING_REQUESTS}" \
  --temperature "${REPLAY_TEMPERATURE}" --top-p 0.95 --top-k 20 \
  --sample-seed "${SAMPLE_SEED}" --sample-ratio "${SAMPLE_RATIO}" --seed 1234 \
  --event-filter "${EVENT_FILTER}" --no-one-event-per-request \
  2>&1 | tee "${OUTPUT_ROOT}/counterfactual_replay.log"

echo "[$(date -Is)] 128-token replay for V4 judge start"
"${PYTHON}" "${HERE}/replay_rejection_branches.py" \
  --trace-dir "${TRACE_ROOT}" \
  --output "${OUTPUT_ROOT}/judge_replay_128.jsonl" \
  --model "${REPLAY_MODEL}" --tp-size "${REPLAY_TP_SIZE}" \
  --cuda-devices "${REPLAY_CUDA_DEVICES}" --mem-fraction-static 0.75 \
  --context-length 96000 --max-new-tokens "${JUDGE_REPLAY_MAX_NEW_TOKENS}" \
  --batch-size "${REPLAY_BATCH_SIZE}" \
  --max-running-requests "${REPLAY_MAX_RUNNING_REQUESTS}" \
  --temperature 0 --top-p 1.0 --top-k -1 \
  --sample-seed "${SAMPLE_SEED}" --sample-ratio "${SAMPLE_RATIO}" --seed 1234 \
  --event-filter "${EVENT_FILTER}" \
  2>&1 | tee "${OUTPUT_ROOT}/judge_replay_128.log"

server_pid=
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET=${NCCL_NET:-Socket}
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

echo "[$(date -Is)] V4 128-token judge server start"
"${PYTHON}" -m sglang.launch_server \
  --model-path "${JUDGE_MODEL}" --tp 8 --dp-size 8 --ep-size 1 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune --swa-full-tokens-ratio 0.2 \
  --chunked-prefill-size 2048 --mem-fraction-static 0.88 \
  --context-length 96000 --cuda-graph-max-bs 64 --max-running-requests 64 \
  --disable-radix-cache --trust-remote-code \
  --host 127.0.0.1 --port "${JUDGE_PORT}" \
  --dist-init-addr 127.0.0.1:30110 --nccl-port 30120 \
  >"${OUTPUT_ROOT}/judge_server.log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + 1800))
until curl -fsS "${JUDGE_BASE_URL%/}/models" >/dev/null; do
  kill -0 "${server_pid}" 2>/dev/null || { echo "Judge server exited" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "Judge startup timed out" >&2; exit 1; }
  sleep 2
done

echo "[$(date -Is)] V4 judge all paired completions start"
"${PYTHON}" "${HERE}/judge_replayed_branches_v4.py" \
  --input "${OUTPUT_ROOT}/judge_replay_128.jsonl" \
  --output "${OUTPUT_ROOT}/judged_v4.jsonl" \
  --base-url "${JUDGE_BASE_URL}" --model "${JUDGE_MODEL}" \
  --parallel-requests "${JUDGE_PARALLEL_REQUESTS}" \
  --timeout 3600 --max-retries 1 --temperature 0 --max-tokens 1536 \
  --seed 42

"${PYTHON}" "${HERE}/summarize_judgments_v3.py" \
  --inputs "${OUTPUT_ROOT}/judged_v4.jsonl" \
  --output-json "${OUTPUT_ROOT}/judge_summary.json" \
  --output-md "${OUTPUT_ROOT}/judge_summary.md"

echo "[$(date -Is)] complete: ${OUTPUT_ROOT}"
