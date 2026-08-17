#!/usr/bin/env bash
# Build a global 5% direct-local sample, launch DeepSeek-V4, judge, and summarize.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/root/sglang-dspark-csd}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
TRACE_ROOT=${TRACE_ROOT:-${HERE}/runs/trace_large_80_30_80_20260814_001505}
TASKS=(lcb_v6 aime25 olympiad_math_en)
OUTPUT_ROOT=${OUTPUT_ROOT:-${HERE}/runs/direct_local_v5_sample5_$(date +%Y%m%d_%H%M%S)}
SAMPLE_RATIO=${SAMPLE_RATIO:-0.05}
SAMPLE_SEED=${SAMPLE_SEED:-42}
MAX_EVENTS=${MAX_EVENTS:-}
RESUME=${RESUME:-0}

JUDGE_MODEL=${JUDGE_MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
JUDGE_PORT=${JUDGE_PORT:-30000}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:${JUDGE_PORT}/v1}
JUDGE_PARALLEL_REQUESTS=${JUDGE_PARALLEL_REQUESTS:-64}

mkdir -p "${OUTPUT_ROOT}"
INPUT=${OUTPUT_ROOT}/direct_local_samples.jsonl
MANIFEST=${OUTPUT_ROOT}/sample_manifest.json
OUTPUT=${OUTPUT_ROOT}/judged_v5_direct_local.jsonl
ERROR_OUTPUT=${OUTPUT_ROOT}/judge_errors.jsonl

if [[ ! -s "${INPUT}" ]]; then
  args=(
    --trace-root "${TRACE_ROOT}"
    --tasks "${TASKS[@]}"
    --output "${INPUT}"
    --manifest "${MANIFEST}"
    --tokenizer /data/model/Qwen3.5-35B-A3B
    --sample-ratio "${SAMPLE_RATIO}"
    --sample-seed "${SAMPLE_SEED}"
  )
  [[ -z "${MAX_EVENTS}" ]] || args+=(--max-events "${MAX_EVENTS}")
  "${PYTHON}" "${HERE}/build_direct_local_samples.py" "${args[@]}"
elif (( ! RESUME )); then
  echo "Refusing to reuse existing input without RESUME=1: ${INPUT}" >&2
  exit 1
fi

cat >"${OUTPUT_ROOT}/config.env" <<EOF
TRACE_ROOT=${TRACE_ROOT}
TASKS=${TASKS[*]}
SAMPLE_RATIO=${SAMPLE_RATIO}
SAMPLE_SEED=${SAMPLE_SEED}
JUDGE_MODEL=${JUDGE_MODEL}
PROMPT_VERSION=V5_5_1_RESOLVE_THEN_COMPARE
CONTEXT=full_prompt_plus_full_generated_prefix
CANDIDATE_SUFFIX_TOKENS=0
EOF

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
  kill -0 "${server_pid}" 2>/dev/null || {
    echo "Direct-local judge server exited; see ${OUTPUT_ROOT}/judge_server.log" >&2
    exit 1
  }
  (( SECONDS < deadline )) || { echo "Judge startup timed out" >&2; exit 1; }
  sleep 2
done

judge_args=(
  --input "${INPUT}"
  --output "${OUTPUT}"
  --error-output "${ERROR_OUTPUT}"
  --base-url "${JUDGE_BASE_URL}"
  --model "${JUDGE_MODEL}"
  --parallel-requests "${JUDGE_PARALLEL_REQUESTS}"
  --timeout 7200
  --max-retries 3
  --temperature 0
  --max-tokens 768
  --seed 42
)
(( RESUME )) && judge_args+=(--resume)
"${PYTHON}" "${HERE}/judge_direct_local.py" "${judge_args[@]}"

"${PYTHON}" "${HERE}/summarize_direct_local.py" \
  --input "${OUTPUT}" \
  --error-input "${ERROR_OUTPUT}" \
  --expected-events "$(wc -l < "${INPUT}")" \
  --output-json "${OUTPUT_ROOT}/summary_v5_direct_local.json" \
  --output-md "${OUTPUT_ROOT}/summary_v5_direct_local.md"

echo "Direct local 5% experiment complete: ${OUTPUT_ROOT}"
