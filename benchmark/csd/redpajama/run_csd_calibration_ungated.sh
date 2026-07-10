#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

OUT_DIR=${OUT_DIR:-/home/zhouxuwen/sglang/benchmark/csd/runs/redpajama}
MODEL_PATH=${MODEL_PATH:-/root/model/Qwen3.6-35B-A3B}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30003}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
TP_SIZE=${TP_SIZE:-4}
SAMPLES_PER_DOMAIN=${SAMPLES_PER_DOMAIN:-1000}
PARALLEL=${PARALLEL:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
SPEC_ALGORITHM=${SPEC_ALGORITHM:-EAGLE}
SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-3}
SPEC_TOPK=${SPEC_TOPK:-1}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-3}
CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-3}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
CSD_RECORD_IGNORE_PROB_RATIO=${CSD_RECORD_IGNORE_PROB_RATIO:-1}
CSD_DELTA_CAPACITY=${CSD_DELTA_CAPACITY:-16777216}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3000}
SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR:-/home/zhouxuwen/sglang/benchmark/csd/runs/profiles}
DATASET_NAME=${DATASET_NAME:-togethercomputer/RedPajama-Data-1T}
DOMAINS=${DOMAINS:-"arxiv c4 common_crawl github stackexchange wikipedia"}
PROMPT_CHARS=${PROMPT_CHARS:-4096}
MIN_PROMPT_CHARS=${MIN_PROMPT_CHARS:-128}

read -r -a DOMAIN_ARGS <<< "${DOMAINS}"
DOMAIN_COUNT=${#DOMAIN_ARGS[@]}

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
RESULT_FILE=${RESULT_FILE:-"${OUT_DIR}/result_redpajama_csd.jsonl"}
safe_filename_part() {
  local value="$1"
  value="${value%/}"
  value="${value##*/}"
  value=$(printf '%s' "${value}" | tr -c 'A-Za-z0-9._-' '-')
  value="${value#-}"
  value="${value%-}"
  if [[ -z "${value}" ]]; then
    value="none"
  fi
  printf '%s' "${value}"
}

MODEL_NAME_PART=$(safe_filename_part "${MODEL_PATH}")
DRAFT_MODEL_NAME=${DRAFT_MODEL_NAME:-mtp}
DRAFT_MODEL_NAME_PART=$(safe_filename_part "${DRAFT_MODEL_NAME}")
SPEC_ALGORITHM_PART=$(safe_filename_part "${SPEC_ALGORITHM}")
SPEC_SHAPE_PART="steps${SPEC_NUM_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS}"
CSD_RECORD_IGNORE_PROB_RATIO_ARGS=()
CSD_RECORD_IGNORE_PROB_RATIO_BENCH_ARGS=()
CSD_TABLE_KIND=logits_gated
if [[ "${CSD_RECORD_IGNORE_PROB_RATIO}" == "1" ]]; then
  CSD_RECORD_IGNORE_PROB_RATIO_ARGS+=(--speculative-csd-dynamic-update-ignore-prob-ratio)
  CSD_RECORD_IGNORE_PROB_RATIO_BENCH_ARGS+=(--csd-dynamic-update-ignore-prob-ratio)
  CSD_TABLE_KIND=logits_ungated
fi
CSD_TABLE_PATH=${CSD_TABLE_PATH:-"${OUT_DIR}/csd_table_redpajama_${CSD_TABLE_KIND}_${DOMAIN_COUNT}domains_n${SAMPLES_PER_DOMAIN}_${MODEL_NAME_PART}_${DRAFT_MODEL_NAME_PART}_${SPEC_ALGORITHM_PART}_${SPEC_SHAPE_PART}_temp${TEMPERATURE}_ratio${CSD_PROB_RATIO}_tp${TP_SIZE}.json"}

SERVER_PID=""
SERVER_PGID=""
SERVER_LOG=""

port_is_open() {
  python - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

kill_port_servers() {
  pkill -f "sglang serve .*--port ${PORT}( |$)" >/dev/null 2>&1 || true
  pkill -f "sglang.launch_server .*--port ${PORT}( |$)" >/dev/null 2>&1 || true
}

wait_for_port_free() {
  local deadline=$((SECONDS + 120))
  while port_is_open; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${HOST}:${PORT} to become free." >&2
      return 1
    fi
    sleep 2
  done
}

wait_for_server() {
  local deadline=$((SECONDS + WATCHDOG_TIMEOUT))
  until curl -fsS "http://${HOST}:${PORT}/health_generate" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "Server exited before becoming ready. Log: ${SERVER_LOG}" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for server. Log: ${SERVER_LOG}" >&2
      return 1
    fi
    sleep 5
  done
}

stop_server() {
  local stopped=0
  local current_pgid=""
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    stopped=1
    current_pgid=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' || true)
    if [[ -n "${SERVER_PGID}" ]] && [[ "${SERVER_PGID}" != "${current_pgid}" ]]; then
      kill -- "-${SERVER_PGID}" >/dev/null 2>&1 || true
    else
      kill "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
  SERVER_PGID=""
  if (( stopped )); then
    kill_port_servers
    wait_for_port_free
  fi
}

cleanup_server() {
  stop_server
  kill_port_servers
  wait_for_port_free || true
}

start_record_server() {
  local run_id="$1"
  SERVER_LOG="${OUT_DIR}/server_record_${run_id}.log"
  echo "Starting RedPajama CSD record server, log: ${SERVER_LOG}"
  kill_port_servers
  wait_for_port_free
  setsid env \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR}" \
    sglang serve \
      --model-path "${MODEL_PATH}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --trust-remote-code \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --mamba-scheduler-strategy extra_buffer \
      --watchdog-timeout "${WATCHDOG_TIMEOUT}" \
      --log-level warning \
      --port "${PORT}" \
      --speculative-algorithm "${SPEC_ALGORITHM}" \
      --speculative-num-steps "${SPEC_NUM_STEPS}" \
      --speculative-eagle-topk "${SPEC_TOPK}" \
      --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
      --speculative-csd \
      --speculative-csd-dynamic-update \
      --speculative-csd-prob-ratio "${CSD_PROB_RATIO}" \
      "${CSD_RECORD_IGNORE_PROB_RATIO_ARGS[@]}" \
      --speculative-csd-delta-capacity "${CSD_DELTA_CAPACITY}" \
      --speculative-csd-force-accept-disabled >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  SERVER_PGID=$(ps -o pgid= -p "${SERVER_PID}" 2>/dev/null | tr -d ' ' || true)
  wait_for_server
}

trap cleanup_server EXIT

run_id="redpajama_${CSD_TABLE_KIND}_tp${TP_SIZE}_temp${TEMPERATURE}_${DOMAIN_COUNT}domains_n${SAMPLES_PER_DOMAIN}"
start_record_server "${run_id}"

python benchmark/csd/redpajama/bench_redpajama_csd.py \
  --dataset-name "${DATASET_NAME}" \
  --domains "${DOMAIN_ARGS[@]}" \
  --samples-per-domain "${SAMPLES_PER_DOMAIN}" \
  --prompt-chars "${PROMPT_CHARS}" \
  --min-prompt-chars "${MIN_PROMPT_CHARS}" \
  --parallel "${PARALLEL}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend srt \
  --answer-file "${OUT_DIR}/redpajama_csd_calibration_answer.jsonl" \
  --result-file "${RESULT_FILE}" \
  --model-name "${MODEL_PATH}" \
  --draft-model-name "${DRAFT_MODEL_NAME}" \
  --run-tag "${run_id}" \
  --speculative-algorithm "${SPEC_ALGORITHM}" \
  --speculative-num-steps "${SPEC_NUM_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
  --csd-enabled \
  --csd-dynamic-update \
  "${CSD_RECORD_IGNORE_PROB_RATIO_BENCH_ARGS[@]}" \
  --csd-force-accept-disabled \
  --csd-freq-threshold "${CSD_FREQ_THRESHOLD}" \
  --csd-prob-ratio "${CSD_PROB_RATIO}" \
  --csd-log-result \
  --csd-save-table-path "${CSD_TABLE_PATH}"

python - <<PY
import json
path = "${CSD_TABLE_PATH}"
payload = json.load(open(path))
entries = payload.get("entries", [])
print({
    "table": path,
    "entries": len(entries),
    "freq_ge_threshold": sum(int(e.get("freq", 0)) >= int("${CSD_FREQ_THRESHOLD}") for e in entries),
    "total_freq": sum(int(e.get("freq", 0)) for e in entries),
})
PY

stop_server
