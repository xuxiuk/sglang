#!/usr/bin/env bash
# Strictly sequential trace -> replay -> judge pipeline for each task.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/root/sglang-dspark-csd}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
RUN_ROOT=${RUN_ROOT:-${HERE}/runs/full_pipeline_$(date +%Y%m%d_%H%M%S)}
TRACE_SCRIPT=${TRACE_SCRIPT:-${HERE}/run_trace_suite.sh}
REPLAY_SCRIPT=${REPLAY_SCRIPT:-${HERE}/replay_rejection_branches.py}
JUDGE_SCRIPT=${JUDGE_SCRIPT:-${HERE}/judge_replayed_branches.py}
SUMMARY_SCRIPT=${SUMMARY_SCRIPT:-${HERE}/summarize_judgments.py}

TASKS=${TASKS:-"lcb_v6 aime25 olympiad_math_en"}
EVENT_FILTER=${EVENT_FILTER:-all}
MAX_REPLAY_EVENTS=${MAX_REPLAY_EVENTS:-0}
REPLAY_MAX_NEW_TOKENS=${REPLAY_MAX_NEW_TOKENS:-256}
REPLAY_BATCH_SIZE=${REPLAY_BATCH_SIZE:-48}
REPLAY_MAX_RUNNING_REQUESTS=${REPLAY_MAX_RUNNING_REQUESTS:-96}
REPLAY_SAMPLE_RATIO=${REPLAY_SAMPLE_RATIO:-0.05}
REPLAY_SAMPLE_SEED=${REPLAY_SAMPLE_SEED:-42}
TRACE_CAPACITY=${TRACE_CAPACITY:-524288}

JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:30000/v1}
JUDGE_MODEL=${JUDGE_MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
JUDGE_API_KEY=${JUDGE_API_KEY:-EMPTY}
JUDGE_PARALLEL_REQUESTS=${JUDGE_PARALLEL_REQUESTS:-64}
JUDGE_SAMPLE_RATIO=${JUDGE_SAMPLE_RATIO:-1.0}
JUDGE_SAMPLE_SEED=${JUDGE_SAMPLE_SEED:-42}
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-3600}
JUDGE_MAX_RETRIES=${JUDGE_MAX_RETRIES:-1}
JUDGE_RESUME=${JUDGE_RESUME:-0}
JUDGE_READY_TIMEOUT=${JUDGE_READY_TIMEOUT:-1800}
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

if [[ -z "${JUDGE_MODEL}" ]]; then
  echo "JUDGE_MODEL must name the model served by JUDGE_BASE_URL" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"

judge_pid=
stop_judge() {
  if [[ -n "${judge_pid}" ]]; then
    kill "${judge_pid}" 2>/dev/null || true
    wait "${judge_pid}" 2>/dev/null || true
    judge_pid=
  fi
}
trap stop_judge EXIT INT TERM

wait_for_judge() {
  local deadline=$((SECONDS + JUDGE_READY_TIMEOUT))
  until curl -fsS -H "Authorization: Bearer ${JUDGE_API_KEY}" \
      "${JUDGE_BASE_URL%/}/models" >/dev/null; do
    if [[ -n "${judge_pid}" ]] && ! kill -0 "${judge_pid}" 2>/dev/null; then
      echo "Judge server exited before becoming ready; see ${RUN_ROOT}/judge_server.log" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Judge service did not become ready: ${JUDGE_BASE_URL}" >&2
      return 1
    fi
    sleep 2
  done
}

start_judge() {
  export CUDA_VISIBLE_DEVICES="${JUDGE_GPU_SET}"
  export NCCL_IB_DISABLE=1
  export NCCL_NET=${NCCL_NET:-Socket}
  export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
  export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
  "${PYTHON}" -m sglang.launch_server \
    --model-path "${JUDGE_MODEL}" \
    --tp "${JUDGE_TP_SIZE}" \
    --dp-size "${JUDGE_DP_SIZE}" \
    --ep-size "${JUDGE_EP_SIZE}" \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-a2a-backend none \
    --moe-runner-backend flashinfer_mxfp4 \
    --disable-flashinfer-autotune \
    --swa-full-tokens-ratio 0.2 \
    --chunked-prefill-size 2048 \
    --mem-fraction-static "${JUDGE_MEM_FRACTION_STATIC}" \
    --context-length "${JUDGE_CONTEXT_LENGTH}" \
    --cuda-graph-max-bs "${JUDGE_MAX_RUNNING_REQUESTS}" \
    --max-running-requests "${JUDGE_MAX_RUNNING_REQUESTS}" \
    --disable-radix-cache \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "${JUDGE_PORT}" \
    --dist-init-addr "${JUDGE_DIST_INIT_ADDR}" \
    --nccl-port "${JUDGE_NCCL_PORT}" \
    >"${RUN_ROOT}/judge_server.log" 2>&1 &
  judge_pid=$!
  wait_for_judge
}

# Phase 1: each task completes trace and replay before the next task starts.
for task in ${TASKS}; do
  task_root="${RUN_ROOT}/${task}"
  trace_stage="${task_root}/trace"
  mkdir -p "${task_root}"
  echo "[$(date -Is)] ${task}: TRACE"
  OUT_DIR="${trace_stage}" RUN_STAMP=unused TRACE_CAPACITY="${TRACE_CAPACITY}" \
    bash "${TRACE_SCRIPT}" "${task}"
  trace_dir="${trace_stage}/${task}/trace"

  echo "[$(date -Is)] ${task}: REPLAY"
  replay_args=(
    --trace-dir "${trace_dir}"
    --output "${task_root}/replay/replayed.jsonl"
    --event-filter "${EVENT_FILTER}"
    --max-new-tokens "${REPLAY_MAX_NEW_TOKENS}"
    --batch-size "${REPLAY_BATCH_SIZE}"
    --max-running-requests "${REPLAY_MAX_RUNNING_REQUESTS}"
    --sample-ratio "${REPLAY_SAMPLE_RATIO}"
    --sample-seed "${REPLAY_SAMPLE_SEED}"
  )
  if (( MAX_REPLAY_EVENTS > 0 )); then
    replay_args+=(--max-events "${MAX_REPLAY_EVENTS}")
  fi
  PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" "${REPLAY_SCRIPT}" "${replay_args[@]}"

  echo "[$(date -Is)] ${task}: TRACE+REPLAY COMPLETE"
done

# Phase 2: all eight GPUs are now free. Start DeepSeek once, then judge tasks
# strictly one by one against the same immutable service.
echo "[$(date -Is)] starting 8-GPU DeepSeek judge"
start_judge
judge_inputs=()
for task in ${TASKS}; do
  task_root="${RUN_ROOT}/${task}"
  echo "[$(date -Is)] ${task}: JUDGE"
  judge_args=(
    --input "${task_root}/replay/replayed.jsonl" \
    --output "${task_root}/judge/judged.jsonl" \
    --base-url "${JUDGE_BASE_URL}" \
    --model "${JUDGE_MODEL}" \
    --api-key "${JUDGE_API_KEY}" \
    --parallel-requests "${JUDGE_PARALLEL_REQUESTS}" \
    --sample-ratio "${JUDGE_SAMPLE_RATIO}" \
    --seed "${JUDGE_SAMPLE_SEED}" \
    --timeout "${JUDGE_TIMEOUT}" \
    --max-retries "${JUDGE_MAX_RETRIES}" \
    --temperature 0
  )
  if (( JUDGE_RESUME )); then
    judge_args+=(--resume)
  fi
  "${PYTHON}" "${JUDGE_SCRIPT}" "${judge_args[@]}"
  judge_inputs+=("${task_root}/judge/judged.jsonl")

  "${PYTHON}" "${SUMMARY_SCRIPT}" \
    --inputs "${judge_inputs[@]}" \
    --output-json "${RUN_ROOT}/summary.json" \
    --output-md "${RUN_ROOT}/summary.md"
  echo "[$(date -Is)] ${task}: COMPLETE"
done
stop_judge

echo "Full sequential pipeline complete: ${RUN_ROOT}"
