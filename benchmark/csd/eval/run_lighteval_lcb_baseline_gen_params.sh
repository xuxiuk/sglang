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
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmark/csd/runs/baseline_sampling_params/${RUN_STAMP}"}
ARTIFACT_DIR=${ARTIFACT_DIR:-"${OUT_DIR}/artifacts"}
RESULT_DIR=${RESULT_DIR:-"${OUT_DIR}/results"}
mkdir -p "${ARTIFACT_DIR}" "${RESULT_DIR}"

MODEL_PATH=${MODEL_PATH:-/home/shared/models/Qwen/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
CUDA_DEVICES=${CUDA_DEVICES:-6,7}
TP_SIZE=${TP_SIZE:-2}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3000}
MAMBA_SCHEDULER_STRATEGY=${MAMBA_SCHEDULER_STRATEGY:-no_buffer}
LIGHTEVAL_PYTHON=${LIGHTEVAL_PYTHON:-python}

LCB_TASK=${LCB_TASK:-lcb:codegeneration_v6}
AIME_TASK=${AIME_TASK:-aime25}
MATH500_TASK=${MATH500_TASK:-minerva_math500}
GSM8K_TASK=${GSM8K_TASK:-gsm8k}
LIMIT=${LIMIT:-}
LCB_MAX_GEN_TOKS=${LCB_MAX_GEN_TOKS:-${MAX_GEN_TOKS:-81920}}
LCB_MAX_LENGTH=${LCB_MAX_LENGTH:-${MAX_LENGTH:-96000}}
AIME_MAX_GEN_TOKS=${AIME_MAX_GEN_TOKS:-${MAX_GEN_TOKS:-81920}}
AIME_MAX_LENGTH=${AIME_MAX_LENGTH:-${MAX_LENGTH:-96000}}
MATH500_MAX_GEN_TOKS=${MATH500_MAX_GEN_TOKS:-${MAX_GEN_TOKS:-81920}}
MATH500_MAX_LENGTH=${MATH500_MAX_LENGTH:-${MAX_LENGTH:-96000}}
GSM8K_MAX_GEN_TOKS=${GSM8K_MAX_GEN_TOKS:-32768}
GSM8K_MAX_LENGTH=${GSM8K_MAX_LENGTH:-40960}
LIGHTEVAL_SAVE_DETAILS=${LIGHTEVAL_SAVE_DETAILS:-1}
LIGHTEVAL_DISABLE_SAMPLE_CACHE=${LIGHTEVAL_DISABLE_SAMPLE_CACHE:-1}
LIGHTEVAL_DATASET_LOADING_PROCESSES=${LIGHTEVAL_DATASET_LOADING_PROCESSES:-1}
LIGHTEVAL_NUM_FEWSHOT_SEEDS=${LIGHTEVAL_NUM_FEWSHOT_SEEDS:-1}
LIGHTEVAL_BOOTSTRAP_ITERS=${LIGHTEVAL_BOOTSTRAP_ITERS:-1000}
LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE=${LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE:-auto}
LIGHTEVAL_REMOVE_REASONING_TAGS=${LIGHTEVAL_REMOVE_REASONING_TAGS:-1}
LIGHTEVAL_REASONING_TAGS="${LIGHTEVAL_REASONING_TAGS:-[('<think>', '</think>')]}"
RESULT_FILE=${RESULT_FILE:-"${RESULT_DIR}/baseline_sampling_params.jsonl"}

CODING_RECOMMENDED_GEN_KWARGS=${CODING_RECOMMENDED_GEN_KWARGS:-temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0}
CODING_TEMP0_GEN_KWARGS=${CODING_TEMP0_GEN_KWARGS:-temperature=0.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0}
CODING_TEMP1_GEN_KWARGS=${CODING_TEMP1_GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0}
THINK_GENERAL_RECOMMENDED_GEN_KWARGS=${THINK_GENERAL_RECOMMENDED_GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0}
THINK_GENERAL_TEMP0_GEN_KWARGS=${THINK_GENERAL_TEMP0_GEN_KWARGS:-temperature=0.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0}
THINK_GENERAL_TEMP1_GEN_KWARGS=${THINK_GENERAL_TEMP1_GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0}
NON_THINK_GENERAL_GEN_KWARGS=${NON_THINK_GENERAL_GEN_KWARGS:-temperature=0.7,top_p=0.8,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0}
NON_THINK_REASONING_GEN_KWARGS=${NON_THINK_REASONING_GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0}
LCB_NON_THINK_GEN_KWARGS=${LCB_NON_THINK_GEN_KWARGS:-${CODING_RECOMMENDED_GEN_KWARGS}}
AIME_NON_THINK_GEN_KWARGS=${AIME_NON_THINK_GEN_KWARGS:-${NON_THINK_REASONING_GEN_KWARGS}}
MATH500_NON_THINK_GEN_KWARGS=${MATH500_NON_THINK_GEN_KWARGS:-${NON_THINK_REASONING_GEN_KWARGS}}
GSM8K_NON_THINK_GEN_KWARGS=${GSM8K_NON_THINK_GEN_KWARGS:-${NON_THINK_REASONING_GEN_KWARGS}}
TASK_FILTER=${TASK_FILTER:-all}
VARIANT_FILTER=${VARIANT_FILTER:-all}

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
  "${LIGHTEVAL_PYTHON}" - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import json
import os
import sys

variant = sys.argv[1]
tasks = sys.argv[2]
gen_kwargs = sys.argv[3]
enable_thinking = sys.argv[4]
max_gen_toks = sys.argv[5]
max_length = sys.argv[6]
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
    "LCB_TASK",
    "AIME_TASK",
    "MATH500_TASK",
    "GSM8K_TASK",
    "LIMIT",
    "LCB_MAX_GEN_TOKS",
    "LCB_MAX_LENGTH",
    "AIME_MAX_GEN_TOKS",
    "AIME_MAX_LENGTH",
    "MATH500_MAX_GEN_TOKS",
    "MATH500_MAX_LENGTH",
    "GSM8K_MAX_GEN_TOKS",
    "GSM8K_MAX_LENGTH",
    "LIGHTEVAL_SAVE_DETAILS",
    "LIGHTEVAL_DISABLE_SAMPLE_CACHE",
    "LIGHTEVAL_DATASET_LOADING_PROCESSES",
    "LIGHTEVAL_NUM_FEWSHOT_SEEDS",
    "LIGHTEVAL_BOOTSTRAP_ITERS",
    "LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE",
    "LIGHTEVAL_REMOVE_REASONING_TAGS",
    "LIGHTEVAL_REASONING_TAGS",
    "CODING_RECOMMENDED_GEN_KWARGS",
    "CODING_TEMP0_GEN_KWARGS",
    "CODING_TEMP1_GEN_KWARGS",
    "THINK_GENERAL_RECOMMENDED_GEN_KWARGS",
    "THINK_GENERAL_TEMP0_GEN_KWARGS",
    "THINK_GENERAL_TEMP1_GEN_KWARGS",
    "NON_THINK_GENERAL_GEN_KWARGS",
    "NON_THINK_REASONING_GEN_KWARGS",
    "LCB_NON_THINK_GEN_KWARGS",
    "AIME_NON_THINK_GEN_KWARGS",
    "MATH500_NON_THINK_GEN_KWARGS",
    "GSM8K_NON_THINK_GEN_KWARGS",
]
config = {
    "mode": "baseline",
    "variant": variant,
    "tasks": tasks,
    "enable_thinking": enable_thinking,
    "gen_kwargs": gen_kwargs,
    "max_gen_toks": max_gen_toks,
    "max_length": max_length,
    "server_log": "in_process_lighteval",
}
for key in keys:
    config[key.lower()] = os.environ.get(key)
print(json.dumps(config, separators=(",", ":")))
PY
}

run_baseline_variant() {
  local variant="$1"
  local tasks="$2"
  local gen_kwargs="$3"
  local enable_thinking="$4"
  local max_gen_toks="$5"
  local max_length="$6"
  local variant_name
  variant_name=$(safe_name "${variant}")
  local lighteval_tasks
  lighteval_tasks=$(lighteval_task_spec "${tasks}")
  local output_path="${ARTIFACT_DIR}/baseline_${variant_name}_lighteval_results.json"
  local metrics_path="${ARTIFACT_DIR}/baseline_${variant_name}_lighteval_metrics.json"
  local tracker_dir="${ARTIFACT_DIR}/baseline_${variant_name}_lighteval_tracker"
  local config_json
  config_json=$(experiment_config_json "${variant}" "${lighteval_tasks}" "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}")

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

  echo "Running baseline variant=${variant}, tasks=${lighteval_tasks}, enable_thinking=${enable_thinking}, max_gen_toks=${max_gen_toks}, gen_kwargs=${gen_kwargs}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    GEN_KWARGS="${gen_kwargs}" \
    "${LIGHTEVAL_PYTHON}" benchmark/csd/eval/run_lighteval_sglang_native.py \
      --model "${MODEL_PATH}" \
      --tokenizer "${TOKENIZER_PATH}" \
      --tasks "${lighteval_tasks}" \
      --max-gen-toks "${max_gen_toks}" \
      --max-length "${max_length}" \
      --trust-remote-code \
      --run-tag "baseline_${variant_name}_${RUN_STAMP}" \
      --mode "baseline" \
      --server-log in_process_lighteval \
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

run_selected_baseline_variant() {
  local task_slug="$1"
  local variant_kind="$2"
  local variant="$3"
  local tasks="$4"
  local gen_kwargs="$5"
  local enable_thinking="$6"
  local max_gen_toks="$7"
  local max_length="$8"

  if [[ "${TASK_FILTER}" != "all" && "${TASK_FILTER}" != "${task_slug}" ]]; then
    return 0
  fi
  if [[ "${VARIANT_FILTER}" != "all" && "${VARIANT_FILTER}" != "${variant_kind}" ]]; then
    return 0
  fi

  run_baseline_variant "${variant}" "${tasks}" "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}"
}

run_task_variants() {
  local task_slug="$1"
  local task_spec="$2"
  local recommended_kwargs="$3"
  local temp0_kwargs="$4"
  local temp1_kwargs="$5"
  local non_think_kwargs="$6"
  local max_gen_toks="$7"
  local max_length="$8"

  run_selected_baseline_variant "${task_slug}" recommended "${task_slug}_thinking_recommended" "${task_spec}" "${recommended_kwargs}" true "${max_gen_toks}" "${max_length}"
  run_selected_baseline_variant "${task_slug}" temp0 "${task_slug}_thinking_temp0" "${task_spec}" "${temp0_kwargs}" true "${max_gen_toks}" "${max_length}"
  run_selected_baseline_variant "${task_slug}" temp1 "${task_slug}_thinking_temp1" "${task_spec}" "${temp1_kwargs}" true "${max_gen_toks}" "${max_length}"
  run_selected_baseline_variant "${task_slug}" nonthinking "${task_slug}_nonthinking" "${task_spec}" "${non_think_kwargs}" false "${max_gen_toks}" "${max_length}"
}

run_task_variants "lcb" "${LCB_TASK}" "${CODING_RECOMMENDED_GEN_KWARGS}" "${CODING_TEMP0_GEN_KWARGS}" "${CODING_TEMP1_GEN_KWARGS}" "${LCB_NON_THINK_GEN_KWARGS}" "${LCB_MAX_GEN_TOKS}" "${LCB_MAX_LENGTH}"
run_task_variants "aime25" "${AIME_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" "${THINK_GENERAL_TEMP0_GEN_KWARGS}" "${THINK_GENERAL_TEMP1_GEN_KWARGS}" "${AIME_NON_THINK_GEN_KWARGS}" "${AIME_MAX_GEN_TOKS}" "${AIME_MAX_LENGTH}"
run_task_variants "math500" "${MATH500_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" "${THINK_GENERAL_TEMP0_GEN_KWARGS}" "${THINK_GENERAL_TEMP1_GEN_KWARGS}" "${MATH500_NON_THINK_GEN_KWARGS}" "${MATH500_MAX_GEN_TOKS}" "${MATH500_MAX_LENGTH}"
run_task_variants "gsm8k" "${GSM8K_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" "${THINK_GENERAL_TEMP0_GEN_KWARGS}" "${THINK_GENERAL_TEMP1_GEN_KWARGS}" "${GSM8K_NON_THINK_GEN_KWARGS}" "${GSM8K_MAX_GEN_TOKS}" "${GSM8K_MAX_LENGTH}"

echo "Done."
echo "OUT_DIR=${OUT_DIR}"
echo "RESULT_FILE=${RESULT_FILE}"
