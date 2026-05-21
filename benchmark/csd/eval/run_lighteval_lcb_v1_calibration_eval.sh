#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-/home/zhouxuwen/sglang/benchmark/csd/runs/lcb_calibration/${RUN_STAMP}}
ARTIFACT_DIR=${ARTIFACT_DIR:-"${OUT_DIR}/artifacts"}
RESULT_DIR=${RESULT_DIR:-"${OUT_DIR}/results"}
TABLE_DIR=${TABLE_DIR:-"${OUT_DIR}/tables"}
mkdir -p "${ARTIFACT_DIR}" "${RESULT_DIR}" "${TABLE_DIR}"

MODEL_PATH=${MODEL_PATH:-/home/shared/models/Qwen/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
DRAFT_MODEL_NAME=${DRAFT_MODEL_NAME:-mtp}
CUDA_DEVICES=${CUDA_DEVICES:-4,5}
TP_SIZE=${TP_SIZE:-2}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-7200}
MAMBA_SCHEDULER_STRATEGY=${MAMBA_SCHEDULER_STRATEGY:-no_buffer}
LIGHTEVAL_PYTHON=${LIGHTEVAL_PYTHON:-python}

CALIBRATION_TASK=${CALIBRATION_TASK:-lcb:codegeneration_v1}
CALIBRATION_TASK_PART=${CALIBRATION_TASK//[^A-Za-z0-9._-]/-}
CALIBRATION_LIMIT=${CALIBRATION_LIMIT:-200}
CALIBRATION_MAX_GEN_TOKS=${CALIBRATION_MAX_GEN_TOKS:-1024}
CALIBRATION_MAX_LENGTH=${CALIBRATION_MAX_LENGTH:-96000}
CALIBRATION_GEN_KWARGS=${CALIBRATION_GEN_KWARGS:-temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0}

EVAL_TASK=${EVAL_TASK:-lcb:codegeneration_v6}
EVAL_LIMIT=${EVAL_LIMIT:-}
EVAL_MAX_GEN_TOKS=${EVAL_MAX_GEN_TOKS:-81920}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-96000}
EVAL_GEN_KWARGS=${EVAL_GEN_KWARGS:-temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0}

LIGHTEVAL_SAVE_DETAILS=${LIGHTEVAL_SAVE_DETAILS:-1}
LIGHTEVAL_DISABLE_SAMPLE_CACHE=${LIGHTEVAL_DISABLE_SAMPLE_CACHE:-1}
LIGHTEVAL_DATASET_LOADING_PROCESSES=${LIGHTEVAL_DATASET_LOADING_PROCESSES:-1}
LIGHTEVAL_NUM_FEWSHOT_SEEDS=${LIGHTEVAL_NUM_FEWSHOT_SEEDS:-1}
LIGHTEVAL_BOOTSTRAP_ITERS=${LIGHTEVAL_BOOTSTRAP_ITERS:-1000}
LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE=${LIGHTEVAL_OVERRIDE_CHAT_TEMPLATE:-auto}
LIGHTEVAL_REASONING_TAGS="${LIGHTEVAL_REASONING_TAGS:-[('<think>', '</think>')]}"

SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-5}
SPEC_TOPK=${SPEC_TOPK:-3}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-15}
CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-6}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}

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
DRAFT_MODEL_NAME_PART=$(safe_filename_part "${DRAFT_MODEL_NAME}")
SPEC_ALGORITHM_PART=$(safe_filename_part "EAGLE")
TABLE_NAME="csd_table_${CALIBRATION_TASK_PART}_n${CALIBRATION_LIMIT}_${MODEL_NAME_PART}_${DRAFT_MODEL_NAME_PART}_${SPEC_ALGORITHM_PART}_temp1.0_${RUN_STAMP}.json"
CSD_TABLE_PATH=${CSD_TABLE_PATH:-"${TABLE_DIR}/${TABLE_NAME}"}
CALIBRATION_RESULT_FILE=${CALIBRATION_RESULT_FILE:-"${RESULT_DIR}/lcb_v1_calibration_result.jsonl"}
EVAL_RESULT_FILE=${EVAL_RESULT_FILE:-"${RESULT_DIR}/lcb_eval_result.jsonl"}

if [[ -e "${CSD_TABLE_PATH}" ]]; then
  echo "Refusing to overwrite existing CSD table: ${CSD_TABLE_PATH}" >&2
  echo "Set CSD_TABLE_PATH to a new filename or RUN_STAMP to a new value." >&2
  exit 1
fi

common_lighteval_args() {
  local task="$1"
  local max_gen_toks="$2"
  local max_length="$3"
  local gen_kwargs="$4"
  local output_dir="$5"
  local output_path="$6"
  local metrics_path="$7"
  local result_jsonl="$8"
  local run_tag="$9"
  local mode="${10}"
  shift 10

  "${LIGHTEVAL_PYTHON}" benchmark/csd/eval/run_lighteval_sglang_native.py \
    --model "${MODEL_PATH}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --tasks "${task}" \
    --max-gen-toks "${max_gen_toks}" \
    --max-length "${max_length}" \
    --trust-remote-code \
    --run-tag "${run_tag}" \
    --mode "${mode}" \
    --server-log in_process_lighteval \
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
    --gen-kwargs "${gen_kwargs}" \
    --output-dir "${output_dir}" \
    --output-path "${output_path}" \
    --metrics-output-path "${metrics_path}" \
    --result-jsonl-path "${result_jsonl}" \
    "$@"
}

run_lighteval() {
  local mode="$1"
  local task="$2"
  local run_id="$3"
  local max_gen_toks="$4"
  local max_length="$5"
  local gen_kwargs="$6"
  local result_jsonl="$7"
  shift 7

  local output_path="${ARTIFACT_DIR}/${mode}_${run_id}_lighteval_results.json"
  local metrics_path="${ARTIFACT_DIR}/${mode}_${run_id}_lighteval_metrics.json"
  local tracker_dir="${ARTIFACT_DIR}/${mode}_${run_id}_lighteval_tracker"
  local extra_args=()
  if [[ "${LIGHTEVAL_SAVE_DETAILS}" == "1" ]]; then
    extra_args+=(--save-details)
  fi
  if [[ "${LIGHTEVAL_DISABLE_SAMPLE_CACHE}" == "1" ]]; then
    extra_args+=(--disable-sample-cache)
  fi

  echo "Running LightEval task=${task}, mode=${mode}, output=${output_path}"
  common_lighteval_args \
    "${task}" \
    "${max_gen_toks}" \
    "${max_length}" \
    "${gen_kwargs}" \
    "${tracker_dir}" \
    "${output_path}" \
    "${metrics_path}" \
    "${result_jsonl}" \
    "${run_id}" \
    "${mode}" \
    "${extra_args[@]}" \
    "$@"
}

CALIBRATION_RUN_ID="calibration_${CALIBRATION_TASK_PART}_n${CALIBRATION_LIMIT}_${RUN_STAMP}_steps${SPEC_NUM_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS}_freq${CSD_FREQ_THRESHOLD}_ratio${CSD_PROB_RATIO}"
run_lighteval "calibration" "${CALIBRATION_TASK}" "${CALIBRATION_RUN_ID}" \
  "${CALIBRATION_MAX_GEN_TOKS}" "${CALIBRATION_MAX_LENGTH}" "${CALIBRATION_GEN_KWARGS}" "${CALIBRATION_RESULT_FILE}" \
  --limit "${CALIBRATION_LIMIT}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "${SPEC_NUM_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
  --csd-enabled \
  --csd-dynamic-update \
  --csd-force-accept-disabled \
  --csd-freq-threshold "${CSD_FREQ_THRESHOLD}" \
  --csd-prob-ratio "${CSD_PROB_RATIO}" \
  --csd-save-table-path "${CSD_TABLE_PATH}"

"${LIGHTEVAL_PYTHON}" - <<PY
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

EVAL_RUN_ID="eval_${EVAL_TASK//[^A-Za-z0-9._-]/-}_from_${CALIBRATION_TASK_PART}_${RUN_STAMP}_steps${SPEC_NUM_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS}_freq${CSD_FREQ_THRESHOLD}_ratio${CSD_PROB_RATIO}"
BASELINE_EXTRA=()
if [[ -n "${EVAL_LIMIT}" ]]; then
  BASELINE_EXTRA+=(--limit "${EVAL_LIMIT}")
fi

run_lighteval "baseline" "${EVAL_TASK}" "baseline_${EVAL_RUN_ID}" \
  "${EVAL_MAX_GEN_TOKS}" "${EVAL_MAX_LENGTH}" "${EVAL_GEN_KWARGS}" "${EVAL_RESULT_FILE}" \
  "${BASELINE_EXTRA[@]}"

run_lighteval "vanilla" "${EVAL_TASK}" "vanilla_${EVAL_RUN_ID}" \
  "${EVAL_MAX_GEN_TOKS}" "${EVAL_MAX_LENGTH}" "${EVAL_GEN_KWARGS}" "${EVAL_RESULT_FILE}" \
  "${BASELINE_EXTRA[@]}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "${SPEC_NUM_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}"

run_lighteval "csd" "${EVAL_TASK}" "csd_${EVAL_RUN_ID}" \
  "${EVAL_MAX_GEN_TOKS}" "${EVAL_MAX_LENGTH}" "${EVAL_GEN_KWARGS}" "${EVAL_RESULT_FILE}" \
  "${BASELINE_EXTRA[@]}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "${SPEC_NUM_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
  --csd-enabled \
  --csd-table-path "${CSD_TABLE_PATH}" \
  --csd-freq-threshold "${CSD_FREQ_THRESHOLD}" \
  --csd-prob-ratio "${CSD_PROB_RATIO}"

echo "Done."
echo "CSD_TABLE_PATH=${CSD_TABLE_PATH}"
echo "CALIBRATION_RESULT_FILE=${CALIBRATION_RESULT_FILE}"
echo "EVAL_RESULT_FILE=${EVAL_RESULT_FILE}"
