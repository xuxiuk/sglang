#!/usr/bin/env bash
# Collect observational MTP rejection traces from code and reasoning tasks.
# CSD table/gates are evaluated, but force acceptance is disabled, so outputs
# follow the unmodified MTP verifier path.
set -euo pipefail

RUN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/root/sglang-dspark-csd}
SOURCE_RUN=${SOURCE_RUN:-${REPO_ROOT}/runs/mtp_csd/qwen35b_mtp314_final}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
MODEL=${MODEL:-/data/model/Qwen3.5-35B-A3B}
TABLE=${TABLE:-${SOURCE_RUN}/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps3_topk1_draft3_temp1.0_ratio0.01.json}
RUNNER=${RUNNER:-${SOURCE_RUN}/vendor/eval/run_lighteval_sglang_native.py}
LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-${SOURCE_RUN}/vendor/lighteval/src}

CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
TP_SIZE=${TP_SIZE:-4}
PORT=${PORT:-32320}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-32768}
MAX_LENGTH=${MAX_LENGTH:-96000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
TRACE_CAPACITY=${TRACE_CAPACITY:-65536}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${RUN_DIR}/runs/trace_suite_${RUN_STAMP}}
TASK_SELECTOR=${1:-all}

LCB_LIMIT=${LCB_LIMIT:-80}
AIME_LIMIT=${AIME_LIMIT:-30}
OLYMPIAD_LIMIT=${OLYMPIAD_LIMIT:-80}
LCB_MAX_GEN_TOKS=${LCB_MAX_GEN_TOKS:-16384}
AIME_MAX_GEN_TOKS=${AIME_MAX_GEN_TOKS:-32768}
OLYMPIAD_MAX_GEN_TOKS=${OLYMPIAD_MAX_GEN_TOKS:-16384}

# Match the generation parameters frozen by the formal Qwen35B code/reasoning run.
GEN_KWARGS=${GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0,seed=1234}

if [[ ! -f "${TABLE}" ]]; then
  echo "CSD table not found: ${TABLE}" >&2
  exit 1
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${REPO_ROOT}/python:${LIGHTEVAL_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

cat >"${OUT_DIR}/config.env" <<EOF
MODEL=${MODEL}
TABLE=${TABLE}
TASKS=lcb:codegeneration_v6|0:${LCB_LIMIT},aime25|0:${AIME_LIMIT},olympiad_bench:OE_TO_maths_en_COMP|0:${OLYMPIAD_LIMIT}
SAMPLE_POLICY=LightEval full-dataset shuffle then truncate; deterministic seed=42
TREE_SHAPE=steps3_topk1_draft4
GEN_KWARGS=${GEN_KWARGS}
CUDA_DEVICES=${CUDA_DEVICES}
TP_SIZE=${TP_SIZE}
PORT=${PORT}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}
LCB_MAX_GEN_TOKS=${LCB_MAX_GEN_TOKS}
AIME_MAX_GEN_TOKS=${AIME_MAX_GEN_TOKS}
OLYMPIAD_MAX_GEN_TOKS=${OLYMPIAD_MAX_GEN_TOKS}
MAX_LENGTH=${MAX_LENGTH}
TRACE_CAPACITY=${TRACE_CAPACITY}
EOF

run_task() {
  local slug=$1
  local task=$2
  local limit=$3
  local max_gen_toks=$4
  local task_dir="${OUT_DIR}/${slug}"
  local trace_dir="${task_dir}/trace"
  mkdir -p "${task_dir}"

  echo "Running ${slug}: task=${task}, limit=${limit}, max_gen=${max_gen_toks}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${RUNNER}" \
    --model "${MODEL}" \
    --tokenizer "${MODEL}" \
    --tasks "${task}" \
    --limit "${limit}" \
    --output-dir "${task_dir}/lighteval_tracker" \
    --output-path "${task_dir}/lighteval_results.json" \
    --metrics-output-path "${task_dir}/sglang_metrics.json" \
    --run-tag "qwen35b_mtp314_rejection_trace_${slug}_n${limit}" \
    --mode csd \
    --server-log "${task_dir}/server.log" \
    --cuda-devices "${CUDA_DEVICES}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --skip-server-warmup \
    --watchdog-timeout 7200 \
    --mamba-scheduler-strategy no_buffer \
    --max-gen-toks "${max_gen_toks}" \
    --max-length "${MAX_LENGTH}" \
    --trust-remote-code \
    --override-chat-template auto \
    --enable-thinking true \
    --gen-kwargs "${GEN_KWARGS}" \
    --save-details \
    --generation-only \
    --force-num-samples 1 \
    --disable-sample-cache \
    --dataset-loading-processes 1 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --csd-enabled \
    --csd-table-path "${TABLE}" \
    --csd-freq-threshold 6 \
    --csd-prob-ratio 0.3 \
    --csd-force-accept-disabled \
    --csd-rejection-trace \
    --csd-rejection-trace-dir "${trace_dir}" \
    --csd-rejection-trace-capacity "${TRACE_CAPACITY}" \
    2>&1 | tee "${task_dir}/run.log"
}

case "${TASK_SELECTOR}" in
  all)
    run_task lcb_v6 'lcb:codegeneration_v6|0' "${LCB_LIMIT}" "${LCB_MAX_GEN_TOKS}"
    run_task aime25 'aime25|0' "${AIME_LIMIT}" "${AIME_MAX_GEN_TOKS}"
    run_task olympiad_math_en 'olympiad_bench:OE_TO_maths_en_COMP|0' "${OLYMPIAD_LIMIT}" "${OLYMPIAD_MAX_GEN_TOKS}"
    ;;
  lcb_v6)
    run_task lcb_v6 'lcb:codegeneration_v6|0' "${LCB_LIMIT}" "${LCB_MAX_GEN_TOKS}"
    ;;
  aime25)
    run_task aime25 'aime25|0' "${AIME_LIMIT}" "${AIME_MAX_GEN_TOKS}"
    ;;
  olympiad_math_en)
    run_task olympiad_math_en 'olympiad_bench:OE_TO_maths_en_COMP|0' "${OLYMPIAD_LIMIT}" "${OLYMPIAD_MAX_GEN_TOKS}"
    ;;
  *)
    echo "Unknown task selector: ${TASK_SELECTOR}" >&2
    exit 2
    ;;
esac

echo "Trace suite complete: ${OUT_DIR}"
