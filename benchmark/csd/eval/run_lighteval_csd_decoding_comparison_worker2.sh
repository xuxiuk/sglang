#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export SGLANG_SRC=${SGLANG_SRC:-"${REPO_ROOT}/python"}
export LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-"${REPO_ROOT}/benchmark/csd/lighteval/src"}
export PYTHONPATH="${SGLANG_SRC}:${LIGHTEVAL_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_decoding_comparison/${RUN_STAMP}"}
ARTIFACT_DIR=${ARTIFACT_DIR:-"${OUT_DIR}/artifacts"}
RESULT_DIR=${RESULT_DIR:-"${OUT_DIR}/results"}
mkdir -p "${ARTIFACT_DIR}" "${RESULT_DIR}"

MODEL_PATH=${MODEL_PATH:-/home/shared/models/Qwen/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
CUDA_DEVICES=${CUDA_DEVICES:-6,7}
TP_SIZE=${TP_SIZE:-2}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-7200}
MAMBA_SCHEDULER_STRATEGY=${MAMBA_SCHEDULER_STRATEGY:-no_buffer}
LIGHTEVAL_PYTHON=${LIGHTEVAL_PYTHON:-python}
LIGHTEVAL_CUSTOM_TASKS=${LIGHTEVAL_CUSTOM_TASKS:-}
LIGHTEVAL_SAVE_DETAILS=${LIGHTEVAL_SAVE_DETAILS:-1}
LIGHTEVAL_DISABLE_SAMPLE_CACHE=${LIGHTEVAL_DISABLE_SAMPLE_CACHE:-1}
LIGHTEVAL_DATASET_LOADING_PROCESSES=${LIGHTEVAL_DATASET_LOADING_PROCESSES:-1}
LIGHTEVAL_NUM_FEWSHOT_SEEDS=${LIGHTEVAL_NUM_FEWSHOT_SEEDS:-1}
LIGHTEVAL_BOOTSTRAP_ITERS=${LIGHTEVAL_BOOTSTRAP_ITERS:-1000}
LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE=${LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE:-auto}
LIGHTEVAL_REMOVE_REASONING_TAGS=${LIGHTEVAL_REMOVE_REASONING_TAGS:-1}
LIGHTEVAL_REASONING_TAGS="${LIGHTEVAL_REASONING_TAGS:-[('<think>', '</think>')]}"
RESULT_FILE=${RESULT_FILE:-"${RESULT_DIR}/csd_decoding_comparison.jsonl"}
SERVER_LOG="in_process_lighteval"

LCB_TASK=${LCB_TASK:-lcb:codegeneration_v6}
AIME_TASK=${AIME_TASK:-aime25}
MATH500_TASK=${MATH500_TASK:-minerva_math500}
GSM8K_TASK=${GSM8K_TASK:-gsm8k}
LIMIT=${LIMIT:-}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
LCB_MAX_GEN_TOKS=${LCB_MAX_GEN_TOKS:-${MAX_GEN_TOKS:-81920}}
LCB_MAX_LENGTH=${LCB_MAX_LENGTH:-${MAX_LENGTH:-96000}}
AIME_MAX_GEN_TOKS=${AIME_MAX_GEN_TOKS:-${MAX_GEN_TOKS:-81920}}
AIME_MAX_LENGTH=${AIME_MAX_LENGTH:-${MAX_LENGTH:-96000}}
MATH500_MAX_GEN_TOKS=${MATH500_MAX_GEN_TOKS:-${MAX_GEN_TOKS:-81920}}
MATH500_MAX_LENGTH=${MATH500_MAX_LENGTH:-${MAX_LENGTH:-96000}}
GSM8K_MAX_GEN_TOKS=${GSM8K_MAX_GEN_TOKS:-32768}
GSM8K_MAX_LENGTH=${GSM8K_MAX_LENGTH:-40960}

CODING_RECOMMENDED_GEN_KWARGS=${CODING_RECOMMENDED_GEN_KWARGS:-temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0}
THINK_GENERAL_RECOMMENDED_GEN_KWARGS=${THINK_GENERAL_RECOMMENDED_GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0}

SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-5}
SPEC_TOPK=${SPEC_TOPK:-1}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-5}
CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-6}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
CSD_TABLE_PROB_RATIO=${CSD_TABLE_PROB_RATIO:-0.3}
CSD_REBUILD_TOP_KEEP=${CSD_REBUILD_TOP_KEEP:-${CSD_REBUILD_TOP_FREQ_RATIO:-15000}}
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
PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/runs/redpajama/csd_table_redpajama_6domains_n${REDPAJAMA_SAMPLES_PER_DOMAIN}_${REDPAJAMA_MODEL_NAME_PART}_${REDPAJAMA_DRAFT_MODEL_NAME_PART}_EAGLE_temp${REDPAJAMA_TEMPERATURE}.json"}
RATIO_CSD_TABLE_PATH=${RATIO_CSD_TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/runs/redpajama/csd_table_redpajama_logits_gated_6domains_n${REDPAJAMA_SAMPLES_PER_DOMAIN}_${REDPAJAMA_MODEL_NAME_PART}_${REDPAJAMA_DRAFT_MODEL_NAME_PART}_EAGLE_${REDPAJAMA_SPEC_SHAPE_PART}_temp${REDPAJAMA_TEMPERATURE}_ratio${CSD_TABLE_PROB_RATIO}.json"}

TASK_FILTER=${TASK_FILTER:-all}
METHOD_FILTER=${METHOD_FILTER:-all}

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

experiment_config_json() {
  "${LIGHTEVAL_PYTHON}" - "$@" <<'PY'
import json
import os
import sys

method, task_slug, task_spec, gen_variant, gen_kwargs, enable_thinking, max_gen_toks, max_length, csd_table_path, dynamic_update, top_keep = sys.argv[1:12]
keys = [
    "RUN_STAMP",
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
    "LIGHTEVAL_SAVE_DETAILS",
    "LIGHTEVAL_DISABLE_SAMPLE_CACHE",
    "LIGHTEVAL_DATASET_LOADING_PROCESSES",
    "LIGHTEVAL_NUM_FEWSHOT_SEEDS",
    "LIGHTEVAL_BOOTSTRAP_ITERS",
    "LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE",
    "LIGHTEVAL_REMOVE_REASONING_TAGS",
    "LIGHTEVAL_REASONING_TAGS",
    "LIMIT",
    "NUM_FEWSHOT",
    "SPEC_NUM_STEPS",
    "SPEC_TOPK",
    "SPEC_DRAFT_TOKENS",
    "CSD_FREQ_THRESHOLD",
    "CSD_PROB_RATIO",
    "CSD_TABLE_PROB_RATIO",
    "CSD_REBUILD_TOP_KEEP",
    "PLAIN_CSD_TABLE_PATH",
    "RATIO_CSD_TABLE_PATH",
]
config = {
    "method": method,
    "mode": "baseline" if method == "baseline" else "csd" if method.startswith("csd_") else "vanilla",
    "task_slug": task_slug,
    "tasks": task_spec,
    "gen_variant": gen_variant,
    "gen_kwargs": gen_kwargs,
    "enable_thinking": enable_thinking,
    "max_gen_toks": max_gen_toks,
    "max_length": max_length,
    "csd_table_path": csd_table_path or None,
    "csd_dynamic_update": dynamic_update == "1",
    "csd_rebuild_top_keep": float(top_keep) if top_keep else None,
    "server_log": "in_process_lighteval",
}
for key in keys:
    config[key.lower()] = os.environ.get(key)
print(json.dumps(config, separators=(",", ":")))
PY
}

run_eval() {
  local method="$1"
  local task_slug="$2"
  local task_spec="$3"
  local gen_variant="$4"
  local gen_kwargs="$5"
  local enable_thinking="$6"
  local max_gen_toks="$7"
  local max_length="$8"
  local csd_table_path="$9"
  local dynamic_update="${10}"
  local top_keep="${11}"
  local mode="vanilla"

  case "${method}" in
    baseline)
      mode="baseline"
      ;;
    vanilla)
      mode="vanilla"
      ;;
    csd_*)
      mode="csd"
      if [[ ! -f "${csd_table_path}" ]]; then
        echo "CSD table not found for method=${method}: ${csd_table_path}" >&2
        return 1
      fi
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 1
      ;;
  esac

  local lighteval_task task_name run_name output_path metrics_path tracker_dir config_json
  lighteval_task=$(lighteval_task_spec "${task_spec}")
  task_name=$(safe_name "${lighteval_task}")
  run_name=$(safe_name "${method}_${task_slug}_${gen_variant}_steps${SPEC_NUM_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS}_freq${CSD_FREQ_THRESHOLD}_ratio${CSD_PROB_RATIO}_tableratio${CSD_TABLE_PROB_RATIO}_topkeep${top_keep:-none}_${RUN_STAMP}")
  output_path="${ARTIFACT_DIR}/${run_name}_lighteval_results.json"
  metrics_path="${ARTIFACT_DIR}/${run_name}_lighteval_metrics.json"
  tracker_dir="${ARTIFACT_DIR}/${run_name}_lighteval_tracker"
  config_json=$(experiment_config_json "${method}" "${task_slug}" "${lighteval_task}" "${gen_variant}" "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "${csd_table_path}" "${dynamic_update}" "${top_keep}")

  local extra_args=()
  if [[ -n "${LIMIT}" ]]; then
    extra_args+=(--limit "${LIMIT}")
  fi
  if [[ "${LIGHTEVAL_SAVE_DETAILS}" == "1" ]]; then
    extra_args+=(--save-details)
  fi
  if [[ "${LIGHTEVAL_DISABLE_SAMPLE_CACHE}" == "1" ]]; then
    extra_args+=(--disable-sample-cache)
  fi
  if [[ "${LIGHTEVAL_REMOVE_REASONING_TAGS}" != "1" ]]; then
    extra_args+=(--keep-reasoning-tags)
  fi
  if [[ -n "${LIGHTEVAL_CUSTOM_TASKS}" ]]; then
    extra_args+=(--custom-tasks "${LIGHTEVAL_CUSTOM_TASKS}")
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
    extra_args+=(
      --csd-enabled
      --csd-table-path "${csd_table_path}"
      --csd-freq-threshold "${CSD_FREQ_THRESHOLD}"
      --csd-prob-ratio "${CSD_PROB_RATIO}"
    )
    if [[ "${dynamic_update}" == "1" ]]; then
      extra_args+=(--csd-dynamic-update)
    fi
    if [[ -n "${top_keep}" ]]; then
      extra_args+=(--csd-rebuild-top-keep "${top_keep}")
    fi
  fi

  echo "Running method=${method}, task=${lighteval_task}, gen_variant=${gen_variant}, enable_thinking=${enable_thinking}, output=${output_path}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    GEN_KWARGS="${gen_kwargs}" \
    "${LIGHTEVAL_PYTHON}" benchmark/csd/eval/run_lighteval_sglang_native.py \
      --model "${MODEL_PATH}" \
      --tokenizer "${TOKENIZER_PATH}" \
      --tasks "${lighteval_task}" \
      --max-gen-toks "${max_gen_toks}" \
      --max-length "${max_length}" \
      --trust-remote-code \
      --run-tag "${run_name}" \
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
      --enable-thinking "${enable_thinking}" \
      --dataset-loading-processes "${LIGHTEVAL_DATASET_LOADING_PROCESSES}" \
      --num-fewshot-seeds "${LIGHTEVAL_NUM_FEWSHOT_SEEDS}" \
      --bootstrap-iters "${LIGHTEVAL_BOOTSTRAP_ITERS}" \
      --reasoning-tags "${LIGHTEVAL_REASONING_TAGS}" \
      --gen-kwargs "${gen_kwargs}" \
      --experiment-config-json "${config_json}" \
      --output-dir "${tracker_dir}" \
      --output-path "${output_path}" \
      --metrics-output-path "${metrics_path}" \
      --result-jsonl-path "${RESULT_FILE}" \
      "${extra_args[@]}"
}

run_method_set() {
  local task_slug="$1"
  local task_spec="$2"
  local gen_kwargs="$3"
  local enable_thinking="$4"
  local max_gen_toks="$5"
  local max_length="$6"
  local methods=(baseline vanilla csd_plain_table_static csd_plain_table csd_plain_table_top_keep csd_ratio_table_static csd_ratio_table csd_ratio_table_top_keep)
  # local methods=(baseline vanilla csd_plain_table_static csd_plain_table)
  local method status

  if [[ "${TASK_FILTER}" != "all" && "${TASK_FILTER}" != "${task_slug}" ]]; then
    return 0
  fi

  for method in "${methods[@]}"; do
    if [[ "${METHOD_FILTER}" != "all" && "${METHOD_FILTER}" != "${method}" ]]; then
      continue
    fi
    status=0
    case "${method}" in
      baseline|vanilla)
        run_eval "${method}" "${task_slug}" "${task_spec}" recommended "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "" 0 "" || status=$?
        ;;
      csd_plain_table_static)
        run_eval "${method}" "${task_slug}" "${task_spec}" recommended "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "${PLAIN_CSD_TABLE_PATH}" 0 "" || status=$?
        ;;
      csd_plain_table)
        run_eval "${method}" "${task_slug}" "${task_spec}" recommended "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "${PLAIN_CSD_TABLE_PATH}" 1 "" || status=$?
        ;;
      csd_plain_table_top_keep)
        run_eval "${method}" "${task_slug}" "${task_spec}" recommended "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "${PLAIN_CSD_TABLE_PATH}" 1 "${CSD_REBUILD_TOP_KEEP}" || status=$?
        ;;
      csd_ratio_table_static)
        run_eval "${method}" "${task_slug}" "${task_spec}" recommended "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "${RATIO_CSD_TABLE_PATH}" 0 "" || status=$?
        ;;
      csd_ratio_table)
        run_eval "${method}" "${task_slug}" "${task_spec}" recommended "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "${RATIO_CSD_TABLE_PATH}" 1 "" || status=$?
        ;;
      csd_ratio_table_top_keep)
        run_eval "${method}" "${task_slug}" "${task_spec}" recommended "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "${RATIO_CSD_TABLE_PATH}" 1 "${CSD_REBUILD_TOP_KEEP}" || status=$?
        ;;
    esac
    if [[ "${status}" != "0" ]]; then
      echo "Run failed and will be skipped: method=${method}, task=${task_slug}, status=${status}" >&2
    fi
  done
}

# run_method_set "lcb" "${LCB_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" true "${LCB_MAX_GEN_TOKS}" "${LCB_MAX_LENGTH}"
run_method_set "aime25" "${AIME_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" true "${AIME_MAX_GEN_TOKS}" "${AIME_MAX_LENGTH}"
run_method_set "math500" "${MATH500_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" true "${MATH500_MAX_GEN_TOKS}" "${MATH500_MAX_LENGTH}"
run_method_set "gsm8k" "${GSM8K_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" true "${GSM8K_MAX_GEN_TOKS}" "${GSM8K_MAX_LENGTH}"

echo "Done."
echo "OUT_DIR=${OUT_DIR}"
echo "RESULT_FILE=${RESULT_FILE}"
