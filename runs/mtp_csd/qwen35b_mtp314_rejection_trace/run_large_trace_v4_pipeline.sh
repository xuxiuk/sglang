#!/usr/bin/env bash
# Reuse the completed 80/30/80 trace, replay a fixed fraction, then run V4 judge.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/root/sglang-dspark-csd}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
TRACE_ROOT=${TRACE_ROOT:-${HERE}/runs/trace_large_80_30_80_20260814_001505}
OUTPUT_ROOT=${OUTPUT_ROOT:-${HERE}/runs/large_v4_sample3_$(date +%Y%m%d_%H%M%S)}
TASKS=${TASKS:-"lcb_v6 aime25 olympiad_math_en"}
SAMPLE_RATIO=${SAMPLE_RATIO:-0.03}
SAMPLE_SEED=${SAMPLE_SEED:-42}
RESUME=${RESUME:-0}

REPLAY_MODEL=${REPLAY_MODEL:-/data/model/Qwen3.5-35B-A3B}
REPLAY_MAX_NEW_TOKENS=${REPLAY_MAX_NEW_TOKENS:-256}
REPLAY_BATCH_SIZE=${REPLAY_BATCH_SIZE:-48}
REPLAY_MAX_RUNNING_REQUESTS=${REPLAY_MAX_RUNNING_REQUESTS:-96}
REPLAY_GPU_SET=${REPLAY_GPU_SET:-0,1,2,3}

JUDGE_MODEL=${JUDGE_MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
JUDGE_PORT=${JUDGE_PORT:-30000}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:${JUDGE_PORT}/v1}
JUDGE_PARALLEL_REQUESTS=${JUDGE_PARALLEL_REQUESTS:-64}

if [[ -e "${OUTPUT_ROOT}" ]] && (( ! RESUME )); then
  echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
for task in ${TASKS}; do
  test -s "${TRACE_ROOT}/${task}/trace/events.rank0.jsonl"
  test -s "${TRACE_ROOT}/${task}/trace/requests.rank0.jsonl"
done

if [[ ! -s "${OUTPUT_ROOT}/config.env" ]]; then
cat >"${OUTPUT_ROOT}/config.env" <<EOF
TRACE_ROOT=${TRACE_ROOT}
TASKS=${TASKS}
SAMPLE_RATIO=${SAMPLE_RATIO}
SAMPLE_SEED=${SAMPLE_SEED}
REPLAY_MAX_NEW_TOKENS=${REPLAY_MAX_NEW_TOKENS}
REPLAY_MODEL=${REPLAY_MODEL}
JUDGE_MODEL=${JUDGE_MODEL}
PROMPT_VERSION=V4_BOUNDARY_CONSTRAINED_INDEPENDENT_VALIDITY
EOF
fi

for task in ${TASKS}; do
  replay_output="${OUTPUT_ROOT}/${task}/replay/replayed.jsonl"
  if [[ -s "${replay_output}" ]] && (( RESUME )); then
    echo "[$(date -Is)] ${task}: REPLAY already complete ($(wc -l < "${replay_output}") events), skipping"
    continue
  fi
  echo "[$(date -Is)] ${task}: 3% REPLAY start"
  PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" "${HERE}/replay_rejection_branches.py" \
      --trace-dir "${TRACE_ROOT}/${task}/trace" \
      --output "${replay_output}" \
      --model "${REPLAY_MODEL}" --tp-size 4 --cuda-devices "${REPLAY_GPU_SET}" \
      --event-filter all --sample-ratio "${SAMPLE_RATIO}" --sample-seed "${SAMPLE_SEED}" \
      --max-new-tokens "${REPLAY_MAX_NEW_TOKENS}" \
      --batch-size "${REPLAY_BATCH_SIZE}" \
      --max-running-requests "${REPLAY_MAX_RUNNING_REQUESTS}"
  echo "[$(date -Is)] ${task}: REPLAY complete ($(wc -l < "${replay_output}") events)"
done

server_pid=
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1
export NCCL_NET=${NCCL_NET:-Socket}
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

"${PYTHON}" -m sglang.launch_server \
  --model-path "${JUDGE_MODEL}" --tp 8 --dp-size 8 --ep-size 1 \
  --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune --swa-full-tokens-ratio 0.2 \
  --chunked-prefill-size 2048 --mem-fraction-static 0.88 \
  --context-length 96000 --cuda-graph-max-bs 64 --max-running-requests 64 \
  --disable-radix-cache --trust-remote-code \
  --host 127.0.0.1 --port "${JUDGE_PORT}" \
  --dist-init-addr 127.0.0.1:30010 --nccl-port 30020 \
  >"${OUTPUT_ROOT}/judge_server.log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + 1800))
until curl -fsS "${JUDGE_BASE_URL%/}/models" >/dev/null; do
  kill -0 "${server_pid}" 2>/dev/null || { echo "V4 judge server exited" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "V4 judge startup timed out" >&2; exit 1; }
  sleep 2
done

outputs=()
for task in ${TASKS}; do
  echo "[$(date -Is)] ${task}: V4 JUDGE start"
  output="${OUTPUT_ROOT}/${task}/judge/judged_v4.jsonl"
  PYTHONPATH="${HERE}:${PYTHONPATH:-}" \
    "${PYTHON}" "${HERE}/judge_replayed_branches_v4.py" \
      --input "${OUTPUT_ROOT}/${task}/replay/replayed.jsonl" \
      --output "${output}" --base-url "${JUDGE_BASE_URL}" --model "${JUDGE_MODEL}" \
      --parallel-requests "${JUDGE_PARALLEL_REQUESTS}" \
      --timeout 3600 --max-retries 3 --temperature 0 --max-tokens 1536 \
      --seed 42 --resume
  outputs+=("${output}")
  echo "[$(date -Is)] ${task}: V4 JUDGE complete"
done

"${PYTHON}" "${HERE}/summarize_judgments_v3.py" \
  --inputs "${outputs[@]}" \
  --output-json "${OUTPUT_ROOT}/summary_v4.json" \
  --output-md "${OUTPUT_ROOT}/summary_v4.md"

"${PYTHON}" "${HERE}/plot_false_rejection_tail.py" \
  --judge "${outputs[@]}" --freq-threshold 6 \
  --output "${OUTPUT_ROOT}/false_rejection_tail_v4.png"

echo "Large-trace V4 pipeline complete: ${OUTPUT_ROOT}"
