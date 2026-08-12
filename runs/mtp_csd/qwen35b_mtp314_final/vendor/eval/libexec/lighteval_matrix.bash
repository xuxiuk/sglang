#!/usr/bin/env bash
# Internal LightEval matrix used by run_mtp515_v0510_six_datasets_mr48.sh.
#
# This script uses the task-recommended "classic" generation parameters:
#   - LCB/code: temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0
#   - AIME/Math500/GSM8K reasoning: temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
#
# Methods:
#   auto             : normal SGLang generation, no EAGLE/CSD
#   eagle            : vanilla EAGLE, CSD off
#   plain            : EAGLE + static plain CSD table
#   dynamic          : EAGLE + plain CSD table + dynamic update
#   dynamic_entropy  : dynamic + force-accept target_entropy gate
#
# Default tree shapes are:
#   313      -> steps=3, topk=1,  draft=3
#   515      -> steps=5, topk=1,  draft=5
#   3-15-5  -> steps=3, topk=15, draft=5
#   3-9-3   -> steps=3, topk=9,  draft=3
#
# Examples:
#   # Smoke test one task/method/shape without running the model:
#   DRY_RUN=1 LIMIT=4 TASK_FILTER=gsm8k METHOD_SET="auto eagle" TREE_SHAPES="313:3:1:3" \
#     bash benchmark/csd/eval/libexec/lighteval_matrix.bash
#
#   # Small real smoke test:
#   LIMIT=8 TASK_FILTER=gsm8k METHOD_SET="auto eagle plain dynamic dynamic_entropy" TREE_SHAPES="313:3:1:3" \
#     bash benchmark/csd/eval/libexec/lighteval_matrix.bash
#
#   # Full default matrix:
#   bash benchmark/csd/eval/libexec/lighteval_matrix.bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT_OVERRIDE:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}
LIGHTEVAL_RUNNER=${LIGHTEVAL_RUNNER:-${SCRIPT_DIR}/../run_lighteval_sglang_native.py}
cd "${REPO_ROOT}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_classic_tree_shape_sweep}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_classic_tree_shape_sweep/${RUN_STAMP}"}
ARTIFACT_DIR=${ARTIFACT_DIR:-"${OUT_ROOT}/artifacts"}
RESULT_DIR=${RESULT_DIR:-"${OUT_ROOT}/results"}
LOG_DIR=${LOG_DIR:-"${OUT_ROOT}/logs"}
RESULT_FILE=${RESULT_FILE:-"${RESULT_DIR}/classic_tree_shape_sweep.jsonl"}
mkdir -p "${ARTIFACT_DIR}" "${RESULT_DIR}" "${LOG_DIR}"

# Prefer the local SGLang source tree, but use the installed sgl-kernel wheel
# produced by `make build` by default. Set USE_SOURCE_SGL_KERNEL=1 only when
# intentionally testing copied .so files from sgl-kernel/build.
export SGLANG_SRC=${SGLANG_SRC:-"${REPO_ROOT}/python"}
export LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-"${REPO_ROOT}/benchmark/csd/lighteval/src"}
USE_SOURCE_SGL_KERNEL=${USE_SOURCE_SGL_KERNEL:-0}
if [[ "${USE_SOURCE_SGL_KERNEL}" == "1" ]]; then
  export SGL_KERNEL_SRC=${SGL_KERNEL_SRC:-"${REPO_ROOT}/sgl-kernel/python"}
  export PYTHONPATH="${SGL_KERNEL_SRC}:${SGLANG_SRC}:${LIGHTEVAL_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${SGLANG_SRC}:${LIGHTEVAL_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"
fi

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export CUDA_HOME
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/root/miniconda3/envs/sglang/bin:${CUDA_HOME}/bin:/home/ccuser/.local/bin:${PATH}"
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

LIGHTEVAL_PYTHON=${LIGHTEVAL_PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/root/model/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
CUDA_DEVICES=${CUDA_DEVICES:-4,5,6,7}
TP_SIZE=${TP_SIZE:-4}
PORT=${PORT:-30001}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-7200}
MAMBA_SCHEDULER_STRATEGY=${MAMBA_SCHEDULER_STRATEGY:-no_buffer}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}
DRY_RUN=${DRY_RUN:-0}
SHOW_KERNEL=${SHOW_KERNEL:-1}
COPY_INCREMENTAL_KERNEL=${COPY_INCREMENTAL_KERNEL:-0}

LIGHTEVAL_CUSTOM_TASKS=${LIGHTEVAL_CUSTOM_TASKS:-}
LIGHTEVAL_SAVE_DETAILS=${LIGHTEVAL_SAVE_DETAILS:-1}
LIGHTEVAL_DISABLE_SAMPLE_CACHE=${LIGHTEVAL_DISABLE_SAMPLE_CACHE:-1}
LIGHTEVAL_DATASET_LOADING_PROCESSES=${LIGHTEVAL_DATASET_LOADING_PROCESSES:-1}
LIGHTEVAL_NUM_FEWSHOT_SEEDS=${LIGHTEVAL_NUM_FEWSHOT_SEEDS:-1}
LIGHTEVAL_BOOTSTRAP_ITERS=${LIGHTEVAL_BOOTSTRAP_ITERS:-1000}
LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE=${LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE:-auto}
LIGHTEVAL_REMOVE_REASONING_TAGS=${LIGHTEVAL_REMOVE_REASONING_TAGS:-1}
LIGHTEVAL_REASONING_TAGS="${LIGHTEVAL_REASONING_TAGS:-[('<think>', '</think>')]}"

LCB_TASK=${LCB_TASK:-lcb:codegeneration_v6}
AIME_TASK=${AIME_TASK:-aime25}
MATH500_TASK=${MATH500_TASK:-minerva_math500}
GSM8K_TASK=${GSM8K_TASK:-gsm8k}
MTBENCH_TASK=${MTBENCH_TASK:-mt_bench}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
LIMIT=${LIMIT:-}
LCB_MAX_GEN_TOKS=${LCB_MAX_GEN_TOKS:-${MAX_GEN_TOKS:-81920}}
LCB_MAX_LENGTH=${LCB_MAX_LENGTH:-${MAX_LENGTH:-96000}}
AIME_MAX_GEN_TOKS=${AIME_MAX_GEN_TOKS:-${MAX_GEN_TOKS:-81920}}
AIME_MAX_LENGTH=${AIME_MAX_LENGTH:-${MAX_LENGTH:-96000}}
MATH500_MAX_GEN_TOKS=${MATH500_MAX_GEN_TOKS:-${MAX_GEN_TOKS:-81920}}
MATH500_MAX_LENGTH=${MATH500_MAX_LENGTH:-${MAX_LENGTH:-96000}}
GSM8K_MAX_GEN_TOKS=${GSM8K_MAX_GEN_TOKS:-32768}
GSM8K_MAX_LENGTH=${GSM8K_MAX_LENGTH:-40960}
MTBENCH_MAX_GEN_TOKS=${MTBENCH_MAX_GEN_TOKS:-1024}
MTBENCH_MAX_LENGTH=${MTBENCH_MAX_LENGTH:-8192}

CODING_RECOMMENDED_GEN_KWARGS=${CODING_RECOMMENDED_GEN_KWARGS:-temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0}
THINK_GENERAL_RECOMMENDED_GEN_KWARGS=${THINK_GENERAL_RECOMMENDED_GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0}
LCB_RECOMMENDED_GEN_KWARGS=${LCB_RECOMMENDED_GEN_KWARGS:-${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}}
LCB_ENABLE_THINKING=${LCB_ENABLE_THINKING:-true}

# Tree item format: label:speculative_num_steps:speculative_eagle_topk:speculative_num_draft_tokens
TREE_SHAPES=${TREE_SHAPES:-"313:3:1:3 515:5:1:5 3-15-5:3:15:5 3-9-3:3:9:3"}
METHOD_SET=${METHOD_SET:-"auto eagle plain dynamic dynamic_entropy"}
TASK_FILTER=${TASK_FILTER:-all}
METHOD_FILTER=${METHOD_FILTER:-all}
TREE_FILTER=${TREE_FILTER:-all}

CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-6}
CSD_KEY_SELECTION_STRATEGY=${CSD_KEY_SELECTION_STRATEGY:-frequency}
CSD_SCORE_THRESHOLD=${CSD_SCORE_THRESHOLD:-0}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
CSD_TABLE_PROB_RATIO=${CSD_TABLE_PROB_RATIO:-0.3}
PLAIN_CSD_TABLE_PROB_RATIO=${PLAIN_CSD_TABLE_PROB_RATIO:-0.01}
CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO=${CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO:-1}
CSD_REBUILD_TOP_KEEP=${CSD_REBUILD_TOP_KEEP:-}
CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD:-4096}
ENTROPY_P20_THRESHOLD=${ENTROPY_P20_THRESHOLD:-1.5638477802276611}
ENTROPY_P30_THRESHOLD=${ENTROPY_P30_THRESHOLD:-1.3415851593017578}
ENTROPY_P40_THRESHOLD=${ENTROPY_P40_THRESHOLD:-1.1563854217529297}
ENTROPY_MIN_P30_THRESHOLD=${ENTROPY_MIN_P30_THRESHOLD:-0.616}
CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD=${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-}

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
PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/assets/calibration/csd_table_redpajama_logits_ungated_6domains_n${REDPAJAMA_SAMPLES_PER_DOMAIN}_${REDPAJAMA_MODEL_NAME_PART}_${REDPAJAMA_DRAFT_MODEL_NAME_PART}_EAGLE_${REDPAJAMA_SPEC_SHAPE_PART}_temp${REDPAJAMA_TEMPERATURE}_ratio${PLAIN_CSD_TABLE_PROB_RATIO}.json"}

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

matches_filter() {
  local value="$1"
  local filter="$2"
  if [[ "${filter}" == "all" ]]; then
    return 0
  fi
  filter=",${filter// /,},"
  [[ "${filter}" == *",${value},"* ]]
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

copy_incremental_kernel() {
  if [[ "${COPY_INCREMENTAL_KERNEL}" != "1" ]]; then
    return 0
  fi

  if [[ -z "${SGL_KERNEL_SRC:-}" ]]; then
    echo "COPY_INCREMENTAL_KERNEL=1 requires SGL_KERNEL_SRC or USE_SOURCE_SGL_KERNEL=1" >&2
    exit 1
  fi

  local build_dir="${REPO_ROOT}/sgl-kernel/build"
  local package_dir="${SGL_KERNEL_SRC}/sgl_kernel"
  local common_src="${build_dir}/sm90/common_ops.abi3.so"
  local common_dst_dir="${package_dir}/sm90"

  if [[ ! -f "${common_src}" ]]; then
    echo "Incremental kernel artifact not found: ${common_src}" >&2
    echo "Run: cmake --build ${build_dir} --target common_ops_sm90_build --parallel 48" >&2
    exit 1
  fi

  mkdir -p "${common_dst_dir}"
  cp "${common_src}" "${common_dst_dir}/common_ops.abi3.so"

  # The source-tree sgl_kernel package shadows any installed wheel. Copy the
  # auxiliary extensions too; otherwise imports such as sgl_kernel.flash_attn
  # fail and SGLang may skip model implementations that depend on FA3.
  local so_name
  for so_name in flash_ops.abi3.so flashmla_ops.abi3.so spatial_ops.abi3.so deep_gemm_cpp.abi3.so; do
    if [[ -f "${build_dir}/${so_name}" ]]; then
      cp "${build_dir}/${so_name}" "${package_dir}/${so_name}"
    fi
  done
}

show_kernel() {
  if [[ "${SHOW_KERNEL}" != "1" ]]; then
    return 0
  fi
  "${LIGHTEVAL_PYTHON}" - <<'PY'
import sgl_kernel
print('sgl_kernel package:', sgl_kernel.__file__)
print('common_ops module:', sgl_kernel.common_ops.__file__)
print('has tree op:', hasattr(sgl_kernel, 'tree_speculative_sampling_target_only'))
PY
}

experiment_config_json() {
  "${LIGHTEVAL_PYTHON}" - "$@" <<'PY'
import json
import os
import sys

(
    method,
    mode,
    task_slug,
    task_spec,
    tree_label,
    steps,
    topk,
    draft,
    gen_kwargs,
    enable_thinking,
    max_gen_toks,
    max_length,
    dynamic_update,
    entropy_threshold,
) = sys.argv[1:15]
keys = [
    "RUN_STAMP",
    "OUT_ROOT",
    "ARTIFACT_DIR",
    "RESULT_DIR",
    "RESULT_FILE",
    "MODEL_PATH",
    "TOKENIZER_PATH",
    "CUDA_DEVICES",
    "TP_SIZE",
    "PORT",
    "MEM_FRACTION_STATIC",
    "MAX_RUNNING_REQUESTS",
    "WATCHDOG_TIMEOUT",
    "MAMBA_SCHEDULER_STRATEGY",
    "LIGHTEVAL_PYTHON",
    "LIMIT",
    "NUM_FEWSHOT",
    "TASK_FILTER",
    "METHOD_SET",
    "TREE_SHAPES",
    "CSD_FREQ_THRESHOLD",
    "CSD_KEY_SELECTION_STRATEGY",
    "CSD_SCORE_THRESHOLD",
    "CSD_PROB_RATIO",
    "CSD_TABLE_PROB_RATIO",
    "PLAIN_CSD_TABLE_PROB_RATIO",
    "PLAIN_CSD_TABLE_PATH",
    "CSD_REBUILD_TOP_KEEP",
    "CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO",
    "ENTROPY_P20_THRESHOLD",
    "ENTROPY_P30_THRESHOLD",
    "ENTROPY_P40_THRESHOLD",
]
config = {
    "method": method,
    "mode": mode,
    "task_slug": task_slug,
    "tasks": task_spec,
    "tree_label": tree_label,
    "speculative_num_steps": int(steps) if steps else None,
    "speculative_eagle_topk": int(topk) if topk else None,
    "speculative_num_draft_tokens": int(draft) if draft else None,
    "gen_variant": "classic_recommended",
    "gen_kwargs": gen_kwargs,
    "enable_thinking": enable_thinking,
    "max_gen_toks": max_gen_toks,
    "max_length": max_length,
    "csd_dynamic_update": dynamic_update == "1",
    "csd_force_accept_entropy_threshold": float(entropy_threshold) if entropy_threshold else None,
    "server_log": "in_process_lighteval",
}
for key in keys:
    config[key.lower()] = os.environ.get(key)
print(json.dumps(config, separators=(",", ":"), ensure_ascii=False))
PY
}

method_config() {
  local method="$1"
  case "${method}" in
    auto|baseline)
      printf 'auto|baseline|0|0||'
      ;;
    eagle|vanilla)
      printf 'eagle|vanilla|0|0||'
      ;;
    plain|static|csd_plain)
      printf 'plain|csd|1|0|||'
      ;;
    plain_entropy_p20|plain+entropy_p20|plain_entropy_ignore_ratio_p20)
      local threshold="${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-${ENTROPY_P20_THRESHOLD}}"
      printf 'plain_entropy_p20|csd|1|0|%s||' "${threshold}"
      ;;
    dynamic|csd_dynamic|dynamic_ratio|dynamic_gate)
      printf 'dynamic|csd|1|1||0|'
      ;;
    dynamic_ignore_ratio|dynamic_ignore|dynamic_nogate)
      printf 'dynamic_ignore_ratio|csd|1|1||1|'
      ;;
    topkeep15000_no_ratio|dynamic_topkeep15000_ignore_ratio)
      printf 'topkeep15000_no_ratio|csd|1|1||1|15000'
      ;;
    dynamic_entropy|dynamic+entropy|entropy|entropy_p30|dynamic_entropy_ratio|dynamic_entropy_gate)
      local threshold="${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-${ENTROPY_P30_THRESHOLD}}"
      printf 'dynamic_entropy|csd|1|1|%s|0|' "${threshold}"
      ;;
    dynamic_entropy_p20|entropy_p20|dynamic+entropy_p20)
      local threshold="${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-${ENTROPY_P20_THRESHOLD}}"
      printf 'dynamic_entropy_p20|csd|1|1|%s|0|' "${threshold}"
      ;;
    dynamic_entropy_p40|entropy_p40|dynamic+entropy_p40)
      local threshold="${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-${ENTROPY_P40_THRESHOLD}}"
      printf 'dynamic_entropy_p40|csd|1|1|%s|0|' "${threshold}"
      ;;
    dynamic_entropy_ignore_ratio|dynamic_entropy_ignore|dynamic_entropy_nogate)
      local threshold="${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-${ENTROPY_P30_THRESHOLD}}"
      printf 'dynamic_entropy_ignore_ratio|csd|1|1|%s|1|' "${threshold}"
      ;;
    dynamic_entropy_p20_ignore_ratio|dynamic_entropy_ignore_ratio_p20|entropy_p20_ignore_ratio)
      local threshold="${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-${ENTROPY_P20_THRESHOLD}}"
      printf 'dynamic_entropy_p20_ignore_ratio|csd|1|1|%s|1|' "${threshold}"
      ;;
    topkeep15000_p20_no_ratio|dynamic_entropy_p20_topkeep15000_ignore_ratio)
      local threshold="${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-${ENTROPY_P20_THRESHOLD}}"
      printf 'topkeep15000_p20_no_ratio|csd|1|1|%s|1|15000' "${threshold}"
      ;;
    dynamic_entropy_p40_ignore_ratio|dynamic_entropy_ignore_ratio_p40|entropy_p40_ignore_ratio)
      local threshold="${CSD_FORCE_ACCEPT_ENTROPY_THRESHOLD:-${ENTROPY_P40_THRESHOLD}}"
      printf 'dynamic_entropy_p40_ignore_ratio|csd|1|1|%s|1|' "${threshold}"
      ;;
    dynamic_entropy_min_p30_ignore_ratio|entropy_min_p30_ignore_ratio)
      printf 'dynamic_entropy_min_p30_ignore_ratio|csd|1|1||1|'
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local tree_label="$1"
  local steps="$2"
  local topk="$3"
  local draft="$4"
  local task_slug="$5"
  local task_spec_raw="$6"
  local gen_kwargs="$7"
  local enable_thinking="$8"
  local max_gen_toks="$9"
  local max_length="${10}"
  local method_requested="${11}"

  local method mode csd_enabled dynamic_update entropy_threshold ignore_prob_ratio method_top_keep cfg
  cfg=$(method_config "${method_requested}") || return 1
  IFS='|' read -r method mode csd_enabled dynamic_update entropy_threshold ignore_prob_ratio method_top_keep <<<"${cfg}"

  if ! matches_filter "${method}" "${METHOD_FILTER}"; then
    return 0
  fi

  local task_spec task_name run_name output_path metrics_path tracker_dir log_path config_json
  task_spec=$(lighteval_task_spec "${task_spec_raw}")
  task_name=$(safe_name "${task_spec}")
  run_name=$(safe_name "${method}_${task_slug}_classic_tree${tree_label}_steps${steps}_topk${topk}_draft${draft}_key${CSD_KEY_SELECTION_STRATEGY}_freq${CSD_FREQ_THRESHOLD}_score${CSD_SCORE_THRESHOLD}_ratio${CSD_PROB_RATIO}_entropy${entropy_threshold:-none}_${RUN_STAMP}")
  output_path="${ARTIFACT_DIR}/${run_name}_lighteval_results.json"
  metrics_path="${ARTIFACT_DIR}/${run_name}_lighteval_metrics.json"
  tracker_dir="${ARTIFACT_DIR}/${run_name}_lighteval_tracker"
  log_path="${LOG_DIR}/${run_name}.log"
  config_json=$(experiment_config_json "${method}" "${mode}" "${task_slug}" "${task_spec}" "${tree_label}" "${steps}" "${topk}" "${draft}" "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "${dynamic_update}" "${entropy_threshold}")

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
  if [[ "${method}" != "auto" ]]; then
    extra_args+=(
      --speculative-algorithm EAGLE
      --speculative-num-steps "${steps}"
      --speculative-eagle-topk "${topk}"
      --speculative-num-draft-tokens "${draft}"
    )
  fi
  if [[ "${csd_enabled}" == "1" ]]; then
    if [[ ! -f "${PLAIN_CSD_TABLE_PATH}" ]]; then
      echo "CSD table not found: ${PLAIN_CSD_TABLE_PATH}" >&2
      return 1
    fi
    extra_args+=(
      --csd-enabled
      --csd-table-path "${PLAIN_CSD_TABLE_PATH}"
      --csd-freq-threshold "${CSD_FREQ_THRESHOLD}"
      --csd-prob-ratio "${CSD_PROB_RATIO}"
    )
    if [[ "${dynamic_update}" == "1" ]]; then
      extra_args+=(
        --csd-dynamic-update
        --csd-rebuild-threshold "${CSD_REBUILD_THRESHOLD}"
      )
      if [[ "${ignore_prob_ratio:-${CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO}}" == "1" ]]; then
        extra_args+=(--csd-dynamic-update-ignore-prob-ratio)
      fi
      local effective_top_keep="${method_top_keep:-${CSD_REBUILD_TOP_KEEP}}"
      if [[ -n "${effective_top_keep}" ]]; then
        extra_args+=(--csd-rebuild-top-keep "${effective_top_keep}")
      fi
    fi
    if [[ -n "${entropy_threshold}" ]]; then
      extra_args+=(--csd-force-accept-entropy-threshold "${entropy_threshold}")
    fi
    if [[ "${method}" == "dynamic_entropy_min_p30_ignore_ratio" ]]; then
      extra_args+=(
        --csd-force-accept-entropy-min-threshold "${ENTROPY_MIN_P30_THRESHOLD}"
      )
    fi
  fi

  local cmd=(
    "${LIGHTEVAL_PYTHON}" "${LIGHTEVAL_RUNNER}"
    --model "${MODEL_PATH}"
    --tokenizer "${TOKENIZER_PATH}"
    --tasks "${task_spec}"
    --max-gen-toks "${max_gen_toks}"
    --max-length "${max_length}"
    --trust-remote-code
    --run-tag "${run_name}"
    --mode "${mode}"
    --server-log in_process_lighteval
    --cuda-devices "${CUDA_DEVICES}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --watchdog-timeout "${WATCHDOG_TIMEOUT}"
    --mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY}"
    --log-level warning
    --override-chat-template "${LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE}"
    --enable-thinking "${enable_thinking}"
    --dataset-loading-processes "${LIGHTEVAL_DATASET_LOADING_PROCESSES}"
    --num-fewshot-seeds "${LIGHTEVAL_NUM_FEWSHOT_SEEDS}"
    --bootstrap-iters "${LIGHTEVAL_BOOTSTRAP_ITERS}"
    --reasoning-tags "${LIGHTEVAL_REASONING_TAGS}"
    --gen-kwargs "${gen_kwargs}"
    --experiment-config-json "${config_json}"
    --output-dir "${tracker_dir}"
    --output-path "${output_path}"
    --metrics-output-path "${metrics_path}"
    --result-jsonl-path "${RESULT_FILE}"
    "${extra_args[@]}"
  )

  echo "------------------------------------------------------------"
  echo "Running method=${method}, task=${task_slug}, tree=${tree_label} (${steps}/${topk}/${draft})"
  echo "  gen_kwargs       : ${gen_kwargs}"
  echo "  enable_thinking  : ${enable_thinking}"
  echo "  entropy_threshold: ${entropy_threshold:-none}"
  echo "  output           : ${output_path}"
  echo "  log              : ${log_path}"
  echo "------------------------------------------------------------"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q GEN_KWARGS=%q ' "${CUDA_DEVICES}" "${gen_kwargs}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  (
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
      GEN_KWARGS="${gen_kwargs}" \
      "${cmd[@]}"
  ) >"${log_path}" 2>&1
}

run_task_methods() {
  local tree_label="$1"
  local steps="$2"
  local topk="$3"
  local draft="$4"
  local task_slug="$5"
  local task_spec="$6"
  local gen_kwargs="$7"
  local enable_thinking="$8"
  local max_gen_toks="$9"
  local max_length="${10}"

  if ! matches_filter "${task_slug}" "${TASK_FILTER}"; then
    return 0
  fi

  local method status
  for method in ${METHOD_SET}; do
    status=0
    run_one "${tree_label}" "${steps}" "${topk}" "${draft}" "${task_slug}" "${task_spec}" \
      "${gen_kwargs}" "${enable_thinking}" "${max_gen_toks}" "${max_length}" "${method}" || status=$?
    if [[ "${status}" != "0" ]]; then
      echo "Run failed: tree=${tree_label}, task=${task_slug}, method=${method}, status=${status}" >&2
      if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
        return "${status}"
      fi
    fi
  done
}

copy_incremental_kernel
show_kernel

cat <<EOF
OUT_ROOT=${OUT_ROOT}
RESULT_FILE=${RESULT_FILE}
LOG_DIR=${LOG_DIR}
MODEL_PATH=${MODEL_PATH}
CUDA_DEVICES=${CUDA_DEVICES}
TP_SIZE=${TP_SIZE}
TREE_SHAPES=${TREE_SHAPES}
METHOD_SET=${METHOD_SET}
TASK_FILTER=${TASK_FILTER}
PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH}
EOF

status=0
for tree_spec in ${TREE_SHAPES}; do
  IFS=: read -r tree_label steps topk draft <<<"${tree_spec}"
  if [[ -z "${tree_label}" || -z "${steps}" || -z "${topk}" || -z "${draft}" ]]; then
    echo "Bad TREE_SHAPES item: ${tree_spec}" >&2
    exit 1
  fi
  if ! matches_filter "${tree_label}" "${TREE_FILTER}"; then
    continue
  fi

  run_task_methods "${tree_label}" "${steps}" "${topk}" "${draft}" \
    "lcb" "${LCB_TASK}" "${LCB_RECOMMENDED_GEN_KWARGS}" "${LCB_ENABLE_THINKING}" "${LCB_MAX_GEN_TOKS}" "${LCB_MAX_LENGTH}" || status=$?
  run_task_methods "${tree_label}" "${steps}" "${topk}" "${draft}" \
    "aime25" "${AIME_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" true "${AIME_MAX_GEN_TOKS}" "${AIME_MAX_LENGTH}" || status=$?
  run_task_methods "${tree_label}" "${steps}" "${topk}" "${draft}" \
    "math500" "${MATH500_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" true "${MATH500_MAX_GEN_TOKS}" "${MATH500_MAX_LENGTH}" || status=$?
  run_task_methods "${tree_label}" "${steps}" "${topk}" "${draft}" \
    "gsm8k" "${GSM8K_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" true "${GSM8K_MAX_GEN_TOKS}" "${GSM8K_MAX_LENGTH}" || status=$?
  run_task_methods "${tree_label}" "${steps}" "${topk}" "${draft}" \
    "mtbench" "${MTBENCH_TASK}" "${THINK_GENERAL_RECOMMENDED_GEN_KWARGS}" true "${MTBENCH_MAX_GEN_TOKS}" "${MTBENCH_MAX_LENGTH}" || status=$?

  if [[ "${status}" != "0" && "${CONTINUE_ON_FAILURE}" != "1" ]]; then
    exit "${status}"
  fi
done

echo "Done."
echo "OUT_ROOT=${OUT_ROOT}"
echo "RESULT_FILE=${RESULT_FILE}"
