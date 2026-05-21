#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

OUT_DIR=${OUT_DIR:-/home/zhouxuwen/sglang/benchmark/csd/runs/lighteval}
MODEL_PATH=${MODEL_PATH:-/home/shared/models/Qwen/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
CUDA_DEVICES=${CUDA_DEVICES:-2,3}
TP_SIZE=${TP_SIZE:-2}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3000}
MAMBA_SCHEDULER_STRATEGY=${MAMBA_SCHEDULER_STRATEGY:-no_buffer}
LIGHTEVAL_PYTHON=${LIGHTEVAL_PYTHON:-python}
LIGHTEVAL_TASKS=${LIGHTEVAL_TASKS:-}
LIGHTEVAL_CUSTOM_TASKS=${LIGHTEVAL_CUSTOM_TASKS:-}
LIGHTEVAL_OUTPUT_DIR=${LIGHTEVAL_OUTPUT_DIR:-}
LIGHTEVAL_SAVE_DETAILS=${LIGHTEVAL_SAVE_DETAILS:-0}
LIGHTEVAL_DISABLE_SAMPLE_CACHE=${LIGHTEVAL_DISABLE_SAMPLE_CACHE:-1}
LIGHTEVAL_DATASET_LOADING_PROCESSES=${LIGHTEVAL_DATASET_LOADING_PROCESSES:-1}
LIGHTEVAL_NUM_FEWSHOT_SEEDS=${LIGHTEVAL_NUM_FEWSHOT_SEEDS:-1}
LIGHTEVAL_BOOTSTRAP_ITERS=${LIGHTEVAL_BOOTSTRAP_ITERS:-1000}
LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE=${LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE:-auto}
LIGHTEVAL_REMOVE_REASONING_TAGS=${LIGHTEVAL_REMOVE_REASONING_TAGS:-1}
LIGHTEVAL_REASONING_TAGS="${LIGHTEVAL_REASONING_TAGS:-[('<think>', '</think>')]}"

THINK_TASKS=${THINK_TASKS:-"aime25 minerva_math500"}
CODING_TASKS=${CODING_TASKS:-"lcb:codegeneration_v6"}
NO_THINK_TASKS=${NO_THINK_TASKS:-"gsm8k"}
TASKS=${TASKS:-"${NO_THINK_TASKS} ${THINK_TASKS} ${CODING_TASKS}"}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
LIMIT=${LIMIT:-}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-81920}
MAX_LENGTH=${MAX_LENGTH:-96000}
GEN_KWARGS=${GEN_KWARGS:-}
RUN_VARIANT=${RUN_VARIANT:-}
THINK_GEN_KWARGS=${THINK_GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0}
CODING_GEN_KWARGS=${CODING_GEN_KWARGS:-temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0}
NO_THINK_GEN_KWARGS=${NO_THINK_GEN_KWARGS:-temperature=1.0,top_p=1.0,top_k=40,min_p=0.0,presence_penalty=2.0,repetition_penalty=1.0}

SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-5}
SPEC_TOPK=${SPEC_TOPK:-3}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-15}
CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-3}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
CSD_TABLE_PROB_RATIO=${CSD_TABLE_PROB_RATIO:-0.3}
CSD_DYNAMIC_UPDATE=${CSD_DYNAMIC_UPDATE:-0}
REDPAJAMA_SAMPLES_PER_DOMAIN=${REDPAJAMA_SAMPLES_PER_DOMAIN:-1000}
REDPAJAMA_TEMPERATURE=${REDPAJAMA_TEMPERATURE:-1.0}
REDPAJAMA_SPEC_NUM_STEPS=${REDPAJAMA_SPEC_NUM_STEPS:-3}
REDPAJAMA_SPEC_TOPK=${REDPAJAMA_SPEC_TOPK:-1}
REDPAJAMA_SPEC_DRAFT_TOKENS=${REDPAJAMA_SPEC_DRAFT_TOKENS:-3}
REDPAJAMA_DRAFT_MODEL_NAME=${REDPAJAMA_DRAFT_MODEL_NAME:-mtp}
REDPAJAMA_MODEL_NAME_PART=${MODEL_PATH%/}
REDPAJAMA_MODEL_NAME_PART=${REDPAJAMA_MODEL_NAME_PART##*/}
REDPAJAMA_MODEL_NAME_PART=$(printf '%s' "${REDPAJAMA_MODEL_NAME_PART}" | tr -c 'A-Za-z0-9._-' '-')
REDPAJAMA_DRAFT_MODEL_NAME_PART=$(printf '%s' "${REDPAJAMA_DRAFT_MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '-')
REDPAJAMA_SPEC_SHAPE_PART="steps${REDPAJAMA_SPEC_NUM_STEPS}_topk${REDPAJAMA_SPEC_TOPK}_draft${REDPAJAMA_SPEC_DRAFT_TOKENS}"
CSD_TABLE_PATH=${CSD_TABLE_PATH:-/home/zhouxuwen/sglang/benchmark/csd/runs/redpajama/csd_table_redpajama_logits_gated_6domains_n${REDPAJAMA_SAMPLES_PER_DOMAIN}_${REDPAJAMA_MODEL_NAME_PART}_${REDPAJAMA_DRAFT_MODEL_NAME_PART}_EAGLE_${REDPAJAMA_SPEC_SHAPE_PART}_temp${REDPAJAMA_TEMPERATURE}_ratio${CSD_TABLE_PROB_RATIO}.json}

ARTIFACT_DIR=${ARTIFACT_DIR:-"${OUT_DIR}/artifacts"}
RESULT_DIR=${RESULT_DIR:-"${OUT_DIR}/results"}
mkdir -p "${ARTIFACT_DIR}" "${RESULT_DIR}"
cd "${REPO_ROOT}"
RESULT_FILE=${RESULT_FILE:-"${RESULT_DIR}/result_baseline.jsonl"}
SERVER_LOG="in_process_lighteval"

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

experiment_config_json() {
  "${LIGHTEVAL_PYTHON}" - "$1" "$2" <<'PY'
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
    "CUDA_DEVICES",
    "TP_SIZE",
    "MEM_FRACTION_STATIC",
    "MAX_RUNNING_REQUESTS",
    "WATCHDOG_TIMEOUT",
    "MAMBA_SCHEDULER_STRATEGY",
    "LIGHTEVAL_PYTHON",
    "LIGHTEVAL_TASKS",
    "LIGHTEVAL_CUSTOM_TASKS",
    "LIGHTEVAL_OUTPUT_DIR",
    "LIGHTEVAL_SAVE_DETAILS",
    "LIGHTEVAL_DISABLE_SAMPLE_CACHE",
    "LIGHTEVAL_DATASET_LOADING_PROCESSES",
    "LIGHTEVAL_NUM_FEWSHOT_SEEDS",
    "LIGHTEVAL_BOOTSTRAP_ITERS",
    "LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE",
    "LIGHTEVAL_REMOVE_REASONING_TAGS",
    "LIGHTEVAL_REASONING_TAGS",
    "TASKS",
    "THINK_TASKS",
    "CODING_TASKS",
    "NO_THINK_TASKS",
    "NUM_FEWSHOT",
    "LIMIT",
    "MAX_GEN_TOKS",
    "MAX_LENGTH",
    "GEN_KWARGS",
    "THINK_GEN_KWARGS",
    "CODING_GEN_KWARGS",
    "NO_THINK_GEN_KWARGS",
    "RUN_VARIANT",
    "SPEC_NUM_STEPS",
    "SPEC_TOPK",
    "SPEC_DRAFT_TOKENS",
    "CSD_FREQ_THRESHOLD",
    "CSD_PROB_RATIO",
    "CSD_DYNAMIC_UPDATE",
    "REDPAJAMA_SAMPLES_PER_DOMAIN",
    "REDPAJAMA_TEMPERATURE",
    "REDPAJAMA_SPEC_NUM_STEPS",
    "REDPAJAMA_SPEC_TOPK",
    "REDPAJAMA_SPEC_DRAFT_TOKENS",
    "CSD_TABLE_PATH",
]
config = {"mode": mode, "server_log": server_log}
for key in keys:
    config[key.lower()] = os.environ.get(key)
print(json.dumps(config, separators=(",", ":")))
PY
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

task_uses_coding() {
  local task="$1"
  local item
  for item in ${CODING_TASKS}; do
    if [[ "${item}" == "${task}" ]]; then
      return 0
    fi
  done
  return 1
}

lighteval_task_spec() {
  local task="$1"
  local fewshot="${NUM_FEWSHOT:-0}"
  if [[ "${task}" == *"|"* ]]; then
    printf '%s' "${task}"
    return 0
  fi
  case "${task}" in
    gsm8k|aime24|aime25|math_500)
      printf '%s|%s' "${task}" "${fewshot}"
      ;;
    minerva_math500)
      printf 'math_500|%s' "${fewshot}"
      ;;
    *)
      printf '%s|%s' "${task}" "${fewshot}"
      ;;
  esac
}

run_lighteval_task() {
  local mode="$1"
  local task="$2"
  local run_id="$3"
  local task_name
  task_name=$(safe_name "${task}")
  local lighteval_task="${LIGHTEVAL_TASKS:-}"
  if [[ -z "${lighteval_task}" ]]; then
    lighteval_task=$(lighteval_task_spec "${task}")
  fi
  task_name=$(safe_name "${lighteval_task}")
  local output_path="${ARTIFACT_DIR}/${mode}_${task_name}_${run_id}_lighteval_results.json"
  local metrics_path="${ARTIFACT_DIR}/${mode}_${task_name}_${run_id}_lighteval_metrics.json"
  local tracker_dir="${LIGHTEVAL_OUTPUT_DIR:-${ARTIFACT_DIR}/${mode}_${task_name}_${run_id}_lighteval_tracker}"
  local config_json
  config_json=$(experiment_config_json "${mode}" "${SERVER_LOG}")

  echo "Running LightEval task=${lighteval_task}, mode=${mode}, output=${output_path}"

  local extra_args=()
  local gen_kwargs
  if [[ -n "${GEN_KWARGS}" ]]; then
    gen_kwargs="${GEN_KWARGS}"
  elif task_uses_coding "${task}"; then
    gen_kwargs="${CODING_GEN_KWARGS}"
  elif task_uses_thinking "${task}"; then
    gen_kwargs="${THINK_GEN_KWARGS}"
  else
    gen_kwargs="${NO_THINK_GEN_KWARGS}"
  fi

  if [[ -n "${LIMIT}" ]]; then
    extra_args+=(--limit "${LIMIT}")
  fi
  if [[ "${LIGHTEVAL_SAVE_DETAILS}" == "1" ]]; then
    extra_args+=(--save-details)
  fi
  if [[ -n "${gen_kwargs}" ]]; then
    extra_args+=(--gen-kwargs "${gen_kwargs}")
  fi
  if [[ -n "${LIGHTEVAL_CUSTOM_TASKS}" ]]; then
    extra_args+=(--custom-tasks "${LIGHTEVAL_CUSTOM_TASKS}")
  fi
  if [[ "${LIGHTEVAL_REMOVE_REASONING_TAGS}" != "1" ]]; then
    extra_args+=(--keep-reasoning-tags)
  fi
  if [[ "${LIGHTEVAL_DISABLE_SAMPLE_CACHE}" == "1" ]]; then
    extra_args+=(--disable-sample-cache)
  fi
  if [[ "${mode}" != "baseline" ]]; then
    extra_args+=(
      --speculative-algorithm EAGLE
      --speculative-num-steps "${SPEC_NUM_STEPS}"
      --speculative-eagle-topk "${SPEC_TOPK}"
      --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}"
    )
  fi
  if [[ "${mode}" == "csd" ]]; then
    extra_args+=(--csd-enabled)
    if [[ "${CSD_DYNAMIC_UPDATE}" == "1" ]]; then
      extra_args+=(--csd-dynamic-update)
    fi
  fi

  GEN_KWARGS="${gen_kwargs}" \
    "${LIGHTEVAL_PYTHON}" benchmark/csd/eval/run_lighteval_sglang_native.py \
    --model "${MODEL_PATH}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --tasks "${lighteval_task}" \
    --max-gen-toks "${MAX_GEN_TOKS}" \
    --max-length "${MAX_LENGTH}" \
    --trust-remote-code \
    --run-tag "${mode}_${task_name}_${run_id}" \
    --mode "${mode}" \
    --server-log "${SERVER_LOG}" \
    --cuda-devices "${CUDA_DEVICES}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --watchdog-timeout "${WATCHDOG_TIMEOUT}" \
    --mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY}" \
    --log-level warning \
    --override-chat-template "${LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE}" \
    --dataset-loading-processes "${LIGHTEVAL_DATASET_LOADING_PROCESSES}" \
    --num-fewshot-seeds "${LIGHTEVAL_NUM_FEWSHOT_SEEDS}" \
    --bootstrap-iters "${LIGHTEVAL_BOOTSTRAP_ITERS}" \
    --reasoning-tags "${LIGHTEVAL_REASONING_TAGS}" \
    --experiment-config-json "${config_json}" \
    --csd-table-path "${CSD_TABLE_PATH}" \
    --csd-freq-threshold "${CSD_FREQ_THRESHOLD}" \
    --csd-prob-ratio "${CSD_PROB_RATIO}" \
    --output-dir "${tracker_dir}" \
    --output-path "${output_path}" \
    --metrics-output-path "${metrics_path}" \
    --result-jsonl-path "${RESULT_FILE}" \
    "${extra_args[@]}"
}

suite_run_id() {
  local value
  value="tasks$(printf '%s' "${TASKS}" | tr ' ' '-')_steps${SPEC_NUM_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS}_freq${CSD_FREQ_THRESHOLD}_ratio${CSD_PROB_RATIO}"
  if [[ -n "${RUN_VARIANT}" ]]; then
    value="${value}_${RUN_VARIANT}"
  fi
  if [[ "${CSD_DYNAMIC_UPDATE}" == "1" ]]; then
    value="${value}_dynamic"
  fi
  safe_name "${value}"
}

normalize_bool() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON)
      printf '1'
      ;;
    0|false|FALSE|no|NO|off|OFF)
      printf '0'
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

apply_suite_overrides() {
  local item key value
  for item in "$@"; do
    if [[ "${item}" != *=* ]]; then
      echo "Unknown suite override: ${item}" >&2
      return 1
    fi
    key="${item%%=*}"
    value="${item#*=}"
    case "${key}" in
      tasks)
        TASKS="${value}"
        LIGHTEVAL_TASKS=""
        ;;
      lighteval_tasks)
        LIGHTEVAL_TASKS="${value}"
        ;;
      gen_kwargs|generation_kwargs)
        GEN_KWARGS="${value}"
        ;;
      temperature|temp)
        GEN_KWARGS="temperature=${value}"
        ;;
      save_details|details)
        LIGHTEVAL_SAVE_DETAILS=$(normalize_bool "${value}")
        ;;
      variant|run_variant|tag)
        RUN_VARIANT="${value}"
        ;;
      steps|spec_num_steps)
        SPEC_NUM_STEPS="${value}"
        ;;
      topk|spec_topk)
        SPEC_TOPK="${value}"
        ;;
      draft|draft_tokens|spec_draft_tokens)
        SPEC_DRAFT_TOKENS="${value}"
        ;;
      threshold|freq|freq_threshold|csd_freq_threshold)
        CSD_FREQ_THRESHOLD="${value}"
        ;;
      ratio|prob_ratio|csd_prob_ratio)
        CSD_PROB_RATIO="${value}"
        ;;
      dynamic|dynamic_update|csd_dynamic_update)
        CSD_DYNAMIC_UPDATE=$(normalize_bool "${value}")
        ;;
      *)
        echo "Unknown suite override: ${key}" >&2
        return 1
        ;;
    esac
  done
}

run_lighteval_suite() {
  local mode="$1"
  shift
  local run_id=""
  if [[ "$#" -gt 0 && "$1" != *=* ]]; then
    run_id="$1"
    shift
  fi

  local old_tasks="${TASKS}"
  local old_lighteval_tasks="${LIGHTEVAL_TASKS}"
  local old_gen_kwargs="${GEN_KWARGS}"
  local old_lighteval_save_details="${LIGHTEVAL_SAVE_DETAILS}"
  local old_run_variant="${RUN_VARIANT}"
  local old_spec_num_steps="${SPEC_NUM_STEPS}"
  local old_spec_topk="${SPEC_TOPK}"
  local old_spec_draft_tokens="${SPEC_DRAFT_TOKENS}"
  local old_csd_freq_threshold="${CSD_FREQ_THRESHOLD}"
  local old_csd_prob_ratio="${CSD_PROB_RATIO}"
  local old_csd_dynamic_update="${CSD_DYNAMIC_UPDATE}"

  if ! apply_suite_overrides "$@"; then
    TASKS="${old_tasks}"
    LIGHTEVAL_TASKS="${old_lighteval_tasks}"
    GEN_KWARGS="${old_gen_kwargs}"
    LIGHTEVAL_SAVE_DETAILS="${old_lighteval_save_details}"
    RUN_VARIANT="${old_run_variant}"
    SPEC_NUM_STEPS="${old_spec_num_steps}"
    SPEC_TOPK="${old_spec_topk}"
    SPEC_DRAFT_TOKENS="${old_spec_draft_tokens}"
    CSD_FREQ_THRESHOLD="${old_csd_freq_threshold}"
    CSD_PROB_RATIO="${old_csd_prob_ratio}"
    CSD_DYNAMIC_UPDATE="${old_csd_dynamic_update}"
    return 1
  fi

  if [[ -z "${run_id}" || "${run_id}" == "auto" ]]; then
    run_id=$(suite_run_id)
  fi

  local status=0
  if [[ -n "${LIGHTEVAL_TASKS}" ]]; then
    run_lighteval_task "${mode}" "${LIGHTEVAL_TASKS}" "${run_id}" || status=$?
  else
    local task
    for task in ${TASKS}; do
      run_lighteval_task "${mode}" "${task}" "${run_id}" || {
        status=$?
        break
      }
    done
  fi

  TASKS="${old_tasks}"
  LIGHTEVAL_TASKS="${old_lighteval_tasks}"
  GEN_KWARGS="${old_gen_kwargs}"
  LIGHTEVAL_SAVE_DETAILS="${old_lighteval_save_details}"
  RUN_VARIANT="${old_run_variant}"
  SPEC_NUM_STEPS="${old_spec_num_steps}"
  SPEC_TOPK="${old_spec_topk}"
  SPEC_DRAFT_TOKENS="${old_spec_draft_tokens}"
  CSD_FREQ_THRESHOLD="${old_csd_freq_threshold}"
  CSD_PROB_RATIO="${old_csd_prob_ratio}"
  CSD_DYNAMIC_UPDATE="${old_csd_dynamic_update}"
  return "${status}"
}

if [[ ! -f "${CSD_TABLE_PATH}" ]]; then
  echo "CSD table not found: ${CSD_TABLE_PATH}" >&2
  echo "Set CSD_TABLE_PATH=/path/to/redpajama_table.json or run benchmark/csd/redpajama/run_csd_calibration.sh first." >&2
  exit 1
fi

# LCB temp=1 / temp=0 comparison with details enabled.
run_lighteval_suite "baseline" auto
run_lighteval_suite "vanilla" auto steps=5 topk=3 draft=15
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp1 GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0" run_lighteval_suite "vanilla" auto steps=5 topk=1 draft=5
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp1 GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0" run_lighteval_suite "csd" auto ratio=0.3 steps=5 topk=3 draft=15
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp1 GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0" run_lighteval_suite "csd" auto dynamic_update=1 ratio=0.3 steps=5 topk=3 draft=15
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp1 GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0" run_lighteval_suite "csd" auto ratio=0.3 steps=5 topk=1 draft=5
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp1 GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0" run_lighteval_suite "csd" auto dynamic_update=1 ratio=0.3 steps=5 topk=1 draft=5

# GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0" run_lighteval_suite "csd" auto ratio=0.3 steps=5 topk=3 draft=15
# GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0" run_lighteval_suite "csd" auto dynamic_update=1 ratio=0.3 steps=5 topk=3 draft=15

# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp0 GEN_KWARGS="temperature=0.0" run_lighteval_suite "baseline" auto
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp0 GEN_KWARGS="temperature=0.0" run_lighteval_suite "vanilla" auto steps=5 topk=3 draft=15
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp0 GEN_KWARGS="temperature=0.0" run_lighteval_suite "vanilla" auto steps=5 topk=1 draft=5
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp0 GEN_KWARGS="temperature=0.0" run_lighteval_suite "csd" auto ratio=0.3 steps=5 topk=3 draft=15
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp0 GEN_KWARGS="temperature=0.0" run_lighteval_suite "csd" auto dynamic_update=1 ratio=0.3 steps=5 topk=3 draft=15
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp0 GEN_KWARGS="temperature=0.0" run_lighteval_suite "csd" auto ratio=0.3 steps=5 topk=1 draft=5
# TASKS="lcb:codegeneration_v6" LIGHTEVAL_SAVE_DETAILS=1 RUN_VARIANT=temp0 GEN_KWARGS="temperature=0.0" run_lighteval_suite "csd" auto dynamic_update=1 ratio=0.3 steps=5 topk=1 draft=5

# Example overrides:
# run_lighteval_suite "csd" auto tasks="gsm8k aime25" dynamic_update=1 threshold=6 ratio=0.01 steps=5 topk=3 draft=15
