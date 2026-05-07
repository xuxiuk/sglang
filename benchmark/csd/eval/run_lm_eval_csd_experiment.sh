#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

OUT_DIR=${OUT_DIR:-/home/zhouxuwen/sglang/benchmark/csd/runs/lm_eval}
MODEL_PATH=${MODEL_PATH:-/home/shared/models/Qwen/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30003}
CUDA_DEVICES=${CUDA_DEVICES:-4,5}
TP_SIZE=${TP_SIZE:-2}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3000}
SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR:-/home/zhouxuwen/sglang/benchmark/csd/runs/profiles}

THINK_TASKS=${THINK_TASKS:-"aime25"}
# NO_THINK_TASKS=${NO_THINK_TASKS:-"gsm8k humaneval minerva_math500"}
NO_THINK_TASKS=${NO_THINK_TASKS:-"gsm8k"}
TASKS=${TASKS:-"${NO_THINK_TASKS} ${THINK_TASKS}"}
NUM_FEWSHOT=${NUM_FEWSHOT:-}
LIMIT=${LIMIT:-1500}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_CONCURRENT=${NUM_CONCURRENT:-8}
PARALLEL_VALUES=${PARALLEL_VALUES:-"1 8 16 32 64 128 256 2048"}
REPEAT_RUNS=${REPEAT_RUNS:-5}
ENABLE_PARALLEL_SWEEP=${ENABLE_PARALLEL_SWEEP:-1}
TIMEOUT=${TIMEOUT:-1800}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-49152}
MAX_LENGTH=${MAX_LENGTH:-96000}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-}
FEWSHOT_AS_MULTITURN=${FEWSHOT_AS_MULTITURN:-}
GEN_KWARGS=${GEN_KWARGS:-}
THINK_APPLY_CHAT_TEMPLATE=${THINK_APPLY_CHAT_TEMPLATE:-1}
NO_THINK_APPLY_CHAT_TEMPLATE=${NO_THINK_APPLY_CHAT_TEMPLATE:-0}
THINK_FEWSHOT_AS_MULTITURN=${THINK_FEWSHOT_AS_MULTITURN:-auto}
NO_THINK_FEWSHOT_AS_MULTITURN=${NO_THINK_FEWSHOT_AS_MULTITURN:-auto}
THINK_GEN_KWARGS=${THINK_GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0}
NO_THINK_GEN_KWARGS=${NO_THINK_GEN_KWARGS:-temperature=0.0}
CONFIRM_RUN_UNSAFE_CODE=${CONFIRM_RUN_UNSAFE_CODE:-1}
HF_ALLOW_CODE_EVAL=${HF_ALLOW_CODE_EVAL:-1}
SAMPLE_LOG=${SAMPLE_LOG:-1}

SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-5}
SPEC_TOPK=${SPEC_TOPK:-3}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-15}
CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-3}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.01}
REDPAJAMA_SAMPLES_PER_DOMAIN=${REDPAJAMA_SAMPLES_PER_DOMAIN:-1000}
REDPAJAMA_TEMPERATURE=${REDPAJAMA_TEMPERATURE:-1.0}
REDPAJAMA_DRAFT_MODEL_NAME=${REDPAJAMA_DRAFT_MODEL_NAME:-mtp}
REDPAJAMA_MODEL_NAME_PART=${MODEL_PATH%/}
REDPAJAMA_MODEL_NAME_PART=${REDPAJAMA_MODEL_NAME_PART##*/}
REDPAJAMA_MODEL_NAME_PART=$(printf '%s' "${REDPAJAMA_MODEL_NAME_PART}" | tr -c 'A-Za-z0-9._-' '-')
REDPAJAMA_DRAFT_MODEL_NAME_PART=$(printf '%s' "${REDPAJAMA_DRAFT_MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '-')
CSD_TABLE_PATH=${CSD_TABLE_PATH:-/home/zhouxuwen/sglang/benchmark/csd/runs/redpajama/csd_table_redpajama_6domains_n${REDPAJAMA_SAMPLES_PER_DOMAIN}_${REDPAJAMA_MODEL_NAME_PART}_${REDPAJAMA_DRAFT_MODEL_NAME_PART}_EAGLE_temp${REDPAJAMA_TEMPERATURE}.json}

ARTIFACT_DIR=${ARTIFACT_DIR:-"${OUT_DIR}/artifacts"}
RESULT_DIR=${RESULT_DIR:-"${OUT_DIR}/results"}
mkdir -p "${ARTIFACT_DIR}" "${RESULT_DIR}"
cd "${REPO_ROOT}"
RESULT_FILE=${RESULT_FILE:-"${RESULT_DIR}/result.jsonl"}

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

start_server() {
  local mode="$1"
  local run_id="$2"
  shift 2
  SERVER_LOG="${ARTIFACT_DIR}/server_${mode}_${run_id}.log"
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
  SERVER_PGID=$(ps -o pgid= -p "${SERVER_PID}" 2>/dev/null | tr -d ' ' || true)
  wait_for_server
}

experiment_config_json() {
  python - "$1" "$2" <<'PY'
import json
import os
import sys

mode = sys.argv[1]
server_log = sys.argv[2]
keys = [
    "OUT_DIR",
    "ARTIFACT_DIR",
    "RESULT_DIR",
    "RESULT_FILE",
    "MODEL_PATH",
    "TOKENIZER_PATH",
    "HOST",
    "PORT",
    "CUDA_DEVICES",
    "TP_SIZE",
    "MEM_FRACTION_STATIC",
    "WATCHDOG_TIMEOUT",
    "SGLANG_TORCH_PROFILER_DIR",
    "TASKS",
    "THINK_TASKS",
    "NO_THINK_TASKS",
    "NUM_FEWSHOT",
    "LIMIT",
    "BATCH_SIZE",
    "NUM_CONCURRENT",
    "PARALLEL_VALUES",
    "REPEAT_RUNS",
    "ENABLE_PARALLEL_SWEEP",
    "CURRENT_PARALLEL",
    "REPEAT_INDEX",
    "SWEEP_ID",
    "TIMEOUT",
    "MAX_GEN_TOKS",
    "MAX_LENGTH",
    "APPLY_CHAT_TEMPLATE",
    "FEWSHOT_AS_MULTITURN",
    "GEN_KWARGS",
    "CONFIRM_RUN_UNSAFE_CODE",
    "HF_ALLOW_CODE_EVAL",
    "SAMPLE_LOG",
    "SPEC_NUM_STEPS",
    "SPEC_TOPK",
    "SPEC_DRAFT_TOKENS",
    "CSD_FREQ_THRESHOLD",
    "CSD_PROB_RATIO",
    "REDPAJAMA_SAMPLES_PER_DOMAIN",
    "REDPAJAMA_TEMPERATURE",
    "CSD_TABLE_PATH",
]
config = {"mode": mode, "server_log": server_log}
for key in keys:
    config[key.lower()] = os.environ.get(key)
print(json.dumps(config, separators=(",", ":")))
PY
}

resolve_num_concurrent() {
  local task="$1"
  local current_parallel="${CURRENT_PARALLEL:-}"
  if [[ -z "${current_parallel}" ]]; then
    printf '%s' "${NUM_CONCURRENT}"
    return 0
  fi
  if [[ "${current_parallel}" != "none" ]]; then
    printf '%s' "${current_parallel}"
    return 0
  fi
  if [[ -n "${LIMIT}" ]]; then
    python - "${LIMIT}" <<'PY'
import math
import sys
print(max(1, math.ceil(float(sys.argv[1]))))
PY
    return 0
  fi
  case "${task}" in
    gsm8k)
      printf '1319'
      ;;
    humaneval)
      printf '164'
      ;;
    minerva_math500|math_500)
      printf '500'
      ;;
    aime24|aime25)
      printf '30'
      ;;
    *)
      echo "Cannot infer all-at-once concurrency for task=${task}; set LIMIT or use a numeric PARALLEL_VALUES entry." >&2
      return 1
      ;;
  esac
}

task_uses_thinking() {
  local task="$1"
  local item
  for item in ${THINK_TASKS}; do
    if [[ "${item}" == "${task}" ]]; then
      return 0
    fi
  done
  return 1
}

run_lm_eval_task() {
  local mode="$1"
  local task="$2"
  local run_id="$3"
  local task_name
  task_name=$(safe_name "${task}")
  local output_path="${ARTIFACT_DIR}/${mode}_${task_name}_${run_id}_results.json"
  local metrics_path="${ARTIFACT_DIR}/${mode}_${task_name}_${run_id}_metrics.json"
  local config_json
  config_json=$(experiment_config_json "${mode}" "${SERVER_LOG}")

  echo "Running lm-eval task=${task}, mode=${mode}, output=${output_path}"

  local extra_args=()
  local apply_chat_template
  local fewshot_as_multiturn
  local gen_kwargs
  if task_uses_thinking "${task}"; then
    apply_chat_template="${THINK_APPLY_CHAT_TEMPLATE}"
    fewshot_as_multiturn="${THINK_FEWSHOT_AS_MULTITURN}"
    gen_kwargs="${THINK_GEN_KWARGS}"
  else
    apply_chat_template="${NO_THINK_APPLY_CHAT_TEMPLATE}"
    fewshot_as_multiturn="${NO_THINK_FEWSHOT_AS_MULTITURN}"
    gen_kwargs="${NO_THINK_GEN_KWARGS}"
  fi

  local resolved_num_concurrent
  resolved_num_concurrent=$(resolve_num_concurrent "${task}")

  if [[ -n "${LIMIT}" ]]; then
    extra_args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${NUM_FEWSHOT}" ]]; then
    extra_args+=(--num-fewshot "${NUM_FEWSHOT}")
  fi
  extra_args+=(--num-concurrent "${resolved_num_concurrent}")
  if [[ -n "${CURRENT_PARALLEL:-}" ]]; then
    extra_args+=(--parallel "${CURRENT_PARALLEL}")
  fi
  if [[ -n "${REPEAT_INDEX:-}" ]]; then
    extra_args+=(--repeat-index "${REPEAT_INDEX}")
  fi
  if [[ -n "${SWEEP_ID:-}" ]]; then
    extra_args+=(--sweep-id "${SWEEP_ID}")
  fi
  if [[ -n "${gen_kwargs}" ]]; then
    extra_args+=(--gen-kwargs "${gen_kwargs}")
  fi
  if [[ "${apply_chat_template}" == "1" ]]; then
    extra_args+=(--apply-chat-template)
  fi
  if [[ "${fewshot_as_multiturn}" != "auto" ]]; then
    extra_args+=(--fewshot-as-multiturn "${fewshot_as_multiturn}")
  fi
  if [[ "${CONFIRM_RUN_UNSAFE_CODE}" == "1" ]]; then
    extra_args+=(--confirm-run-unsafe-code)
  fi
  if [[ "${SAMPLE_LOG}" == "1" ]]; then
    extra_args+=(--sample-log-path "${ARTIFACT_DIR}/${mode}_${task_name}_${run_id}_samples.jsonl")
  fi

  APPLY_CHAT_TEMPLATE="${apply_chat_template}" \
    FEWSHOT_AS_MULTITURN="${fewshot_as_multiturn}" \
    GEN_KWARGS="${gen_kwargs}" \
    HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL}" \
    python benchmark/csd/eval/run_lm_eval_sglang_native.py \
    --base-url "http://${HOST}:${PORT}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --tasks "${task}" \
    --batch-size "${BATCH_SIZE}" \
    --timeout "${TIMEOUT}" \
    --max-gen-toks "${MAX_GEN_TOKS}" \
    --max-length "${MAX_LENGTH}" \
    --trust-remote-code \
    --run-tag "${mode}_${task_name}_${run_id}" \
    --mode "${mode}" \
    --server-log "${SERVER_LOG}" \
    --cuda-devices "${CUDA_DEVICES}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --watchdog-timeout "${WATCHDOG_TIMEOUT}" \
    --experiment-config-json "${config_json}" \
    --speculative-algorithm "$([[ "${mode}" == "baseline" ]] && echo none || echo EAGLE)" \
    --speculative-num-steps "${SPEC_NUM_STEPS}" \
    --speculative-eagle-topk "${SPEC_TOPK}" \
    --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
    --csd-table-path "${CSD_TABLE_PATH}" \
    --csd-freq-threshold "${CSD_FREQ_THRESHOLD}" \
    --csd-prob-ratio "${CSD_PROB_RATIO}" \
    --output-path "${output_path}" \
    --metrics-output-path "${metrics_path}" \
    --result-jsonl-path "${RESULT_FILE}" \
    "${extra_args[@]}"
}

run_lm_eval_suite() {
  local mode="$1"
  local run_id="$2"
  for task in ${TASKS}; do
    run_lm_eval_task "${mode}" "${task}" "${run_id}"
  done
}

trap cleanup_server EXIT

if [[ ! -f "${CSD_TABLE_PATH}" ]]; then
  echo "CSD table not found: ${CSD_TABLE_PATH}" >&2
  echo "Set CSD_TABLE_PATH=/path/to/redpajama_table.json or run benchmark/csd/redpajama/run_csd_calibration.sh first." >&2
  exit 1
fi

run_id="tasks$(echo "${TASKS}" | tr ' ' '-')_steps${SPEC_NUM_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS}_freq${CSD_FREQ_THRESHOLD}_ratio${CSD_PROB_RATIO}"
run_id=$(safe_name "${run_id}")

run_all_modes_once() {
  local run_id="$1"

  stop_server
  start_server "baseline" "${run_id}"
  run_lm_eval_suite "baseline" "${run_id}"

  stop_server
  start_server "vanilla" "${run_id}" \
    --speculative-algorithm EAGLE \
    --speculative-num-steps "${SPEC_NUM_STEPS}" \
    --speculative-eagle-topk "${SPEC_TOPK}" \
    --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}"
  run_lm_eval_suite "vanilla" "${run_id}"

  stop_server
  start_server "csd" "${run_id}" \
    --speculative-algorithm EAGLE \
    --speculative-num-steps "${SPEC_NUM_STEPS}" \
    --speculative-eagle-topk "${SPEC_TOPK}" \
    --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
    --speculative-csd \
    --speculative-csd-table-path "${CSD_TABLE_PATH}" \
    --speculative-csd-freq-threshold "${CSD_FREQ_THRESHOLD}" \
    --speculative-csd-prob-ratio "${CSD_PROB_RATIO}"
  run_lm_eval_suite "csd" "${run_id}"

  stop_server
}

write_sweep_summary() {
  local summary_file="${RESULT_DIR}/parallel_summary.jsonl"
  python - "${RESULT_FILE}" "${summary_file}" "${SWEEP_ID:-}" <<'PY'
import json
import statistics
import sys
from collections import defaultdict

result_file, summary_file, sweep_id = sys.argv[1], sys.argv[2], sys.argv[3]
groups = defaultdict(list)
with open(result_file, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("parallel") is None:
            continue
        if sweep_id and row.get("sweep_id") != sweep_id:
            continue
        key = (row.get("task"), row.get("mode") or row.get("other", {}).get("mode"), str(row.get("parallel")))
        groups[key].append(row)

with open(summary_file, "w", encoding="utf-8") as f:
    for (task, mode, parallel), rows in sorted(groups):
        throughputs = [r.get("throughput") for r in rows if isinstance(r.get("throughput"), (int, float))]
        latencies = [r.get("latency") for r in rows if isinstance(r.get("latency"), (int, float))]
        request_throughputs = [
            r.get("other", {}).get("performance", {}).get("request_throughput")
            for r in rows
            if isinstance(r.get("other", {}).get("performance", {}).get("request_throughput"), (int, float))
        ]
        out = {
            "sweep_id": sweep_id or None,
            "task": task,
            "mode": mode,
            "parallel": parallel,
            "runs": len(rows),
            "mean_throughput": round(statistics.mean(throughputs), 6) if throughputs else None,
            "stdev_throughput": round(statistics.stdev(throughputs), 6) if len(throughputs) > 1 else 0,
            "mean_latency": round(statistics.mean(latencies), 6) if latencies else None,
            "mean_request_throughput": round(statistics.mean(request_throughputs), 6) if request_throughputs else None,
        }
        f.write(json.dumps(out) + "\n")
print(f"Parallel sweep summary written to {summary_file}")
PY
}

if [[ "${ENABLE_PARALLEL_SWEEP}" == "1" ]]; then
  SWEEP_ID=${SWEEP_ID:-"parallel_${run_id}"}
  for CURRENT_PARALLEL in ${PARALLEL_VALUES}; do
    for REPEAT_INDEX in $(seq 1 "${REPEAT_RUNS}"); do
      echo "Running sweep parallel=${CURRENT_PARALLEL}, repeat=${REPEAT_INDEX}/${REPEAT_RUNS}"
      sweep_run_id=$(safe_name "${run_id}_parallel${CURRENT_PARALLEL}_repeat${REPEAT_INDEX}")
      export CURRENT_PARALLEL REPEAT_INDEX SWEEP_ID
      run_all_modes_once "${sweep_run_id}"
    done
  done
  write_sweep_summary
else
  CURRENT_PARALLEL=""
  REPEAT_INDEX=""
  SWEEP_ID=""
  export CURRENT_PARALLEL REPEAT_INDEX SWEEP_ID
  run_all_modes_once "${run_id}"
fi
