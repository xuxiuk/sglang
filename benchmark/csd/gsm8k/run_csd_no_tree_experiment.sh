#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

OUT_DIR=${OUT_DIR:-/home/zhouxuwen/sglang/benchmark/csd/runs/gsm8k_no_tree}
DATA_PATH=${DATA_PATH:-/home/zhouxuwen/sglang/benchmark/gsm8k/test.jsonl}
MODEL_PATH=${MODEL_PATH:-/home/shared/models/Qwen/Qwen3.5-35B-A3B}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30001}
CUDA_DEVICES=${CUDA_DEVICES:-0,4}
TP_SIZE=${TP_SIZE:-2}
NUM_QUESTIONS=${NUM_QUESTIONS:-1500}
NUM_SHOTS=${NUM_SHOTS:-5}
PARALLEL_LIST=${PARALLEL_LIST:-"1 2 4 8 16"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TOP_P=${TOP_P:-1.0}
TEMPERATURES=${TEMPERATURES:-"0.0 1.0"}
SPEC_LENGTHS=${SPEC_LENGTHS:-"4 8 12 15"}
SPEC_TOPK=${SPEC_TOPK:-1}
CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-3}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.01}
REDPAJAMA_SAMPLES_PER_DOMAIN=${REDPAJAMA_SAMPLES_PER_DOMAIN:-1000}
REDPAJAMA_TEMPERATURE=${REDPAJAMA_TEMPERATURE:-1.0}
REDPAJAMA_DRAFT_MODEL_NAME=${REDPAJAMA_DRAFT_MODEL_NAME:-mtp}
REDPAJAMA_MODEL_NAME_PART=${MODEL_PATH%/}
REDPAJAMA_MODEL_NAME_PART=${REDPAJAMA_MODEL_NAME_PART##*/}
REDPAJAMA_MODEL_NAME_PART=$(printf '%s' "${REDPAJAMA_MODEL_NAME_PART}" | tr -c 'A-Za-z0-9._-' '-')
REDPAJAMA_DRAFT_MODEL_NAME_PART=$(printf '%s' "${REDPAJAMA_DRAFT_MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '-')
CSD_TABLE_PATH=${CSD_TABLE_PATH:-/home/zhouxuwen/sglang/benchmark/csd/runs/redpajama/csd_table_redpajama_logits_gated_6domains_n${REDPAJAMA_SAMPLES_PER_DOMAIN}_${REDPAJAMA_MODEL_NAME_PART}_${REDPAJAMA_DRAFT_MODEL_NAME_PART}_EAGLE_temp${REDPAJAMA_TEMPERATURE}_ratio${CSD_PROB_RATIO}.json}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3000}
SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR:-/home/zhouxuwen/sglang/benchmark/csd/runs/profiles}

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
RESULT_FILE=${RESULT_FILE:-"${OUT_DIR}/result_no_tree.jsonl"}

SERVER_PID=""
SERVER_PGID=""
SERVER_LOG=""

safe_name() {
  local value="$1"
  value="${value%/}"
  value="${value##*/}"
  value="${value//[^A-Za-z0-9._-]/-}"
  value="${value##[-_]}"
  value="${value%%[-_]}"
  if [[ -z "${value}" ]]; then
    value="none"
  fi
  printf '%s' "${value}"
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

cleanup_server() {
  stop_server
  kill_port_servers
  wait_for_port_free || true
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

start_server() {
  local mode="$1"
  local run_id="$2"
  shift 2
  SERVER_LOG="${OUT_DIR}/server_${mode}_${run_id}.log"
  echo "Starting ${mode} server, log: ${SERVER_LOG}"
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
      "$@" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  SERVER_PGID=$(ps -o pgid= -p "${SERVER_PID}" | tr -d ' ')
  wait_for_server
}

run_bench() {
  local run_tag="$1"
  local temp="$2"
  local run_id="$3"
  local parallel="$4"
  shift 4

  python benchmark/gsm8k/bench_sglang_eagle.py \
    --data-path "${DATA_PATH}" \
    --num-questions "${NUM_QUESTIONS}" \
    --num-shots "${NUM_SHOTS}" \
    --parallel "${parallel}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${temp}" \
    --top-p "${TOP_P}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --backend srt \
    --answer-file "${OUT_DIR}/${run_tag}_gsm8k_n${NUM_QUESTIONS}_${run_id}_answer.txt" \
    --raw-result-file "${OUT_DIR}/${run_tag}_gsm8k_n${NUM_QUESTIONS}_${run_id}_raw.jsonl" \
    --result-file "${RESULT_FILE}" \
    --dataset-name gsm8k \
    --model-name "${MODEL_PATH}" \
    --draft-model-name mtp \
    --run-tag "${run_tag}_${run_id}" \
    "$@"
}

trap cleanup_server EXIT

if [[ ! -f "${CSD_TABLE_PATH}" ]]; then
  echo "CSD table not found: ${CSD_TABLE_PATH}" >&2
  echo "Set CSD_TABLE_PATH=/path/to/redpajama_table.json or run benchmark/csd/redpajama/run_csd_calibration.sh first." >&2
  exit 1
fi

for temp in ${TEMPERATURES}; do
  temp_tag=${temp//./p}
  for parallel in ${PARALLEL_LIST}; do
    baseline_run_id="temp${temp_tag}_parallel${parallel}_baseline"

    stop_server
    start_server "baseline" "${baseline_run_id}"
    run_bench "baseline" "${temp}" "${baseline_run_id}" "${parallel}" \
      --speculative-algorithm none

    for spec_len in ${SPEC_LENGTHS}; do
      run_id="temp${temp_tag}_parallel${parallel}_len${spec_len}_topk${SPEC_TOPK}"

      echo "=== temperature=${temp}, parallel=${parallel}, spec_len=${spec_len}, csd_table=${CSD_TABLE_PATH} ==="

      stop_server
      start_server "vanilla" "${run_id}" \
        --speculative-algorithm EAGLE \
        --speculative-num-steps "${spec_len}" \
        --speculative-eagle-topk "${SPEC_TOPK}" \
        --speculative-num-draft-tokens "${spec_len}"
      run_bench "vanilla" "${temp}" "${run_id}" "${parallel}" \
        --speculative-algorithm EAGLE \
        --speculative-num-steps "${spec_len}" \
        --speculative-eagle-topk "${SPEC_TOPK}" \
        --speculative-num-draft-tokens "${spec_len}"

      stop_server
      start_server "csd" "${run_id}" \
        --speculative-algorithm EAGLE \
        --speculative-num-steps "${spec_len}" \
        --speculative-eagle-topk "${SPEC_TOPK}" \
        --speculative-num-draft-tokens "${spec_len}" \
        --max-running-requests 1 \
        --speculative-csd \
        --speculative-csd-table-path "${CSD_TABLE_PATH}" \
        --speculative-csd-freq-threshold "${CSD_FREQ_THRESHOLD}" \
        --speculative-csd-prob-ratio "${CSD_PROB_RATIO}"
      run_bench "csd" "${temp}" "${run_id}" "${parallel}" \
        --speculative-algorithm EAGLE \
        --speculative-num-steps "${spec_len}" \
        --speculative-eagle-topk "${SPEC_TOPK}" \
        --speculative-num-draft-tokens "${spec_len}" \
        --csd-enabled \
        --csd-table-path "${CSD_TABLE_PATH}" \
        --csd-freq-threshold "${CSD_FREQ_THRESHOLD}" \
        --csd-prob-ratio "${CSD_PROB_RATIO}" \
        --csd-log-result
    done
  done
done

stop_server

python - <<PY
import json, os
path = "${RESULT_FILE}"
if not os.path.exists(path):
    raise SystemExit(0)
for line in open(path):
    row = json.loads(line)
    other = row.get("other", {})
    print({
        "run_tag": other.get("run_tag"),
        "temperature": other.get("temperature"),
        "accuracy": row.get("accuracy"),
        "invalid": row.get("invalid"),
        "throughput": row.get("throughput"),
        "accept_length": row.get("accept_length"),
        "spec_success_rate": row.get("spec_success_rate"),
        "csd_lookup_hit_ct": other.get("csd_lookup_hit_ct"),
        "csd_forced_accept_ct": other.get("csd_forced_accept_ct"),
        "csd_delta_pair_ct": other.get("csd_delta_pair_ct"),
        "parallel": other.get("parallel"),
        "speculative": other.get("speculative"),
    })
PY
