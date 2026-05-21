#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
CALIBRATION_SCRIPT="${REPO_ROOT}/benchmark/csd/redpajama/run_csd_calibration.sh"
NATIVE_EVAL="${REPO_ROOT}/benchmark/csd/eval/run_lighteval_sglang_native.py"

export SGLANG_SRC=${SGLANG_SRC:-"${REPO_ROOT}/python"}
export LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-"${REPO_ROOT}/benchmark/csd/lighteval/src"}
export PYTHONPATH="${SGLANG_SRC}:${LIGHTEVAL_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ROOT=${RUN_ROOT:-"${REPO_ROOT}/benchmark/csd/runs/debug_three_tables"}
CALIB_DIR=${CALIB_DIR:-"${RUN_ROOT}/calibration"}
EVAL_DIR=${EVAL_DIR:-"${RUN_ROOT}/eval"}
DEBUG_DIR=${DEBUG_DIR:-"${RUN_ROOT}/debug_events"}
mkdir -p "${CALIB_DIR}" "${EVAL_DIR}" "${DEBUG_DIR}"

MODEL_PATH=${MODEL_PATH:-/home/shared/models/Qwen/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-"${MODEL_PATH}"}
CUDA_DEVICES=${CUDA_DEVICES:-4,5}
TP_SIZE=${TP_SIZE:-2}
CALIB_PORT=${CALIB_PORT:-30002}

SAMPLES_PER_DOMAIN=${SAMPLES_PER_DOMAIN:-1000}
CALIB_PARALLEL=${CALIB_PARALLEL:-8}
CALIB_MAX_NEW_TOKENS=${CALIB_MAX_NEW_TOKENS:-512}
CALIB_TEMPERATURE=${CALIB_TEMPERATURE:-1.0}
CALIB_TOP_P=${CALIB_TOP_P:-1.0}
CALIB_DOMAINS=${CALIB_DOMAINS:-"arxiv c4 common_crawl github stackexchange wikipedia"}
CALIB_PROMPT_CHARS=${CALIB_PROMPT_CHARS:-4096}
CALIB_MIN_PROMPT_CHARS=${CALIB_MIN_PROMPT_CHARS:-128}

SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-5}
SPEC_TOPK=${SPEC_TOPK:-3}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-15}
CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-3}
CSD_DELTA_CAPACITY=${CSD_DELTA_CAPACITY:-16777216}

LCB_TASK=${LCB_TASK:-"lcb:codegeneration_v6"}
TASKS=${TASKS:-"${LCB_TASK}"}
LIMIT=${LIMIT:-}
GEN_KWARGS=${GEN_KWARGS:-"temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0"}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-81920}
MAX_LENGTH=${MAX_LENGTH:-96000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3000}
DATASET_LOADING_PROCESSES=${DATASET_LOADING_PROCESSES:-1}
SAVE_DETAILS=${SAVE_DETAILS:-1}

CSD_DEBUG_SAMPLE_RATE=${CSD_DEBUG_SAMPLE_RATE:-100}
CSD_DEBUG_EVENT_CAPACITY=${CSD_DEBUG_EVENT_CAPACITY:-1048576}
SKIP_EXISTING_TABLES=${SKIP_EXISTING_TABLES:-1}
RUN_CALIBRATION=${RUN_CALIBRATION:-1}
RUN_EVAL=${RUN_EVAL:-1}
RUN_RATIOS=${RUN_RATIOS:-"0 0.3 1"}

ratio_slug() {
  local value="$1"
  if [[ "${value}" == "0" || "${value}" == "0.0" ]]; then
    printf 'nogate'
  else
    printf 'ratio%s' "$(printf '%s' "${value}" | tr '.' 'p')"
  fi
}

run_calibration_variant() {
  local ratio="$1"
  local slug="$2"
  local table_path="$3"
  local calib_debug_path="${DEBUG_DIR}/calibration_${slug}.jsonl"

  if [[ "${RUN_CALIBRATION}" != "1" ]]; then
    echo "Skipping calibration for ${slug}; RUN_CALIBRATION=${RUN_CALIBRATION}"
    return 0
  fi
  if [[ "${SKIP_EXISTING_TABLES}" == "1" && -s "${table_path}" ]]; then
    echo "Skipping existing table for ${slug}: ${table_path}"
    return 0
  fi
  rm -f "${table_path}" "${calib_debug_path}"

  echo "Calibrating ${slug}: ratio=${ratio}, table=${table_path}"
  OUT_DIR="${CALIB_DIR}/${slug}" \
  MODEL_PATH="${MODEL_PATH}" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  TP_SIZE="${TP_SIZE}" \
  PORT="${CALIB_PORT}" \
  SAMPLES_PER_DOMAIN="${SAMPLES_PER_DOMAIN}" \
  PARALLEL="${CALIB_PARALLEL}" \
  MAX_NEW_TOKENS="${CALIB_MAX_NEW_TOKENS}" \
  TEMPERATURE="${CALIB_TEMPERATURE}" \
  TOP_P="${CALIB_TOP_P}" \
  DOMAINS="${CALIB_DOMAINS}" \
  PROMPT_CHARS="${CALIB_PROMPT_CHARS}" \
  MIN_PROMPT_CHARS="${CALIB_MIN_PROMPT_CHARS}" \
  SPEC_NUM_STEPS="${SPEC_NUM_STEPS}" \
  SPEC_TOPK="${SPEC_TOPK}" \
  SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS}" \
  CSD_FREQ_THRESHOLD="${CSD_FREQ_THRESHOLD}" \
  CSD_PROB_RATIO="${ratio}" \
  CSD_DELTA_CAPACITY="${CSD_DELTA_CAPACITY}" \
  CSD_TABLE_PATH="${table_path}" \
  CSD_DEBUG_STATS=1 \
  CSD_DEBUG_EVENT_CAPACITY="${CSD_DEBUG_EVENT_CAPACITY}" \
  CSD_DEBUG_SAMPLE_RATE="${CSD_DEBUG_SAMPLE_RATE}" \
  CSD_DEBUG_SAVE_PATH="${calib_debug_path}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT}" \
  bash "${CALIBRATION_SCRIPT}"
}

run_eval_variant() {
  local ratio="$1"
  local slug="$2"
  local table_path="$3"
  local variant_eval_dir="${EVAL_DIR}/${slug}"
  local sample_output_path="${variant_eval_dir}/samples.jsonl"
  local force_debug_path="${DEBUG_DIR}/force_accept_${slug}.jsonl"
  mkdir -p "${variant_eval_dir}"
  rm -f "${force_debug_path}"

  if [[ "${RUN_EVAL}" != "1" ]]; then
    echo "Skipping eval for ${slug}; RUN_EVAL=${RUN_EVAL}"
    return 0
  fi
  if [[ ! -s "${table_path}" ]]; then
    echo "Missing table for ${slug}: ${table_path}" >&2
    return 1
  fi

  local limit_args=()
  if [[ -n "${LIMIT}" ]]; then
    limit_args+=(--limit "${LIMIT}")
  fi
  local detail_args=()
  if [[ "${SAVE_DETAILS}" == "1" ]]; then
    detail_args+=(--save-details)
  fi

  echo "Evaluating ${slug}: ratio=${ratio}, force_debug=${force_debug_path}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  python "${NATIVE_EVAL}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --tasks "${TASKS}" \
    "${limit_args[@]}" \
    --output-dir "${variant_eval_dir}/tracker" \
    --output-path "${variant_eval_dir}/result.json" \
    --metrics-output-path "${variant_eval_dir}/metrics.json" \
    --result-jsonl-path "${EVAL_DIR}/result_${slug}.jsonl" \
    --sample-output-jsonl-path "${sample_output_path}" \
    --run-tag "three_table_debug_${slug}" \
    --mode "csd_debug_${slug}" \
    --cuda-devices "${CUDA_DEVICES}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --watchdog-timeout "${WATCHDOG_TIMEOUT}" \
    --max-gen-toks "${MAX_GEN_TOKS}" \
    --max-length "${MAX_LENGTH}" \
    --trust-remote-code \
    --gen-kwargs "${GEN_KWARGS}" \
    --dataset-loading-processes "${DATASET_LOADING_PROCESSES}" \
    --speculative-algorithm EAGLE \
    --speculative-num-steps "${SPEC_NUM_STEPS}" \
    --speculative-eagle-topk "${SPEC_TOPK}" \
    --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
    --csd-enabled \
    --csd-table-path "${table_path}" \
    --csd-freq-threshold "${CSD_FREQ_THRESHOLD}" \
    --csd-prob-ratio "${ratio}" \
    --csd-debug-stats \
    --csd-debug-event-capacity "${CSD_DEBUG_EVENT_CAPACITY}" \
    --csd-debug-sample-rate "${CSD_DEBUG_SAMPLE_RATE}" \
    --csd-debug-save-path "${force_debug_path}" \
    "${detail_args[@]}"
}

python - <<'PY'
import importlib.util
import os
import sglang
import sgl_kernel
print({
    "sglang": getattr(sglang, "__file__", None),
    "sgl_kernel": getattr(sgl_kernel, "__file__", None),
    "SGLANG_SRC": os.environ.get("SGLANG_SRC"),
    "LIGHTEVAL_SRC": os.environ.get("LIGHTEVAL_SRC"),
})
PY

for ratio in ${RUN_RATIOS}; do
  slug=$(ratio_slug "${ratio}")
  table_path="${CALIB_DIR}/csd_table_redpajama_debug_${slug}_n${SAMPLES_PER_DOMAIN}_steps${SPEC_NUM_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS}_temp${CALIB_TEMPERATURE}_ratio${ratio}.json"
  run_calibration_variant "${ratio}" "${slug}" "${table_path}"
  run_eval_variant "${ratio}" "${slug}" "${table_path}"
  echo "Finished ${slug}"
  echo "  table: ${table_path}"
  echo "  calibration debug JSONL: ${DEBUG_DIR}/calibration_${slug}.jsonl"
  echo "  sample outputs JSONL: ${EVAL_DIR}/${slug}/samples.jsonl"
  echo "  force-accept debug JSONL: ${DEBUG_DIR}/force_accept_${slug}.jsonl"
done

echo "All outputs are under ${RUN_ROOT}"
