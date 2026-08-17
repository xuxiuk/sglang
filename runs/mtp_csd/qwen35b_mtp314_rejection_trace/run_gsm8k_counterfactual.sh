#!/usr/bin/env bash
# Collect a full GSM8K MTP rejection trace, then measure one-token causal effects.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/root/sglang-dspark-csd}
SOURCE_RUN=${SOURCE_RUN:-${REPO_ROOT}/runs/mtp_csd/qwen35b_mtp314_final}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
MODEL=${MODEL:-/data/model/Qwen3.5-35B-A3B}
TABLE=${TABLE:-${SOURCE_RUN}/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps3_topk1_draft3_temp1.0_ratio0.01.json}
RUNNER=${RUNNER:-${SOURCE_RUN}/vendor/eval/run_lighteval_sglang_native.py}
LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-${SOURCE_RUN}/vendor/lighteval/src}

CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
TP_SIZE=${TP_SIZE:-4}
PORT=${PORT:-32340}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
MAX_LENGTH=${MAX_LENGTH:-96000}
GSM8K_LIMIT=${GSM8K_LIMIT:-1319}
GSM8K_MAX_GEN_TOKS=${GSM8K_MAX_GEN_TOKS:-8192}
TRACE_CAPACITY=${TRACE_CAPACITY:-1048576}
GEN_KWARGS=${GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0,seed=1234}

COUNTERFACTUAL_MAX_EVENTS=${COUNTERFACTUAL_MAX_EVENTS:-256}
COUNTERFACTUAL_MAX_NEW_TOKENS=${COUNTERFACTUAL_MAX_NEW_TOKENS:-8192}
COUNTERFACTUAL_BATCH_SIZE=${COUNTERFACTUAL_BATCH_SIZE:-32}
COUNTERFACTUAL_MAX_RUNNING_REQUESTS=${COUNTERFACTUAL_MAX_RUNNING_REQUESTS:-64}
COUNTERFACTUAL_TEMPERATURE=${COUNTERFACTUAL_TEMPERATURE:-0.0}
COUNTERFACTUAL_EVENT_FILTER=${COUNTERFACTUAL_EVENT_FILTER:-would-force-accept}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${HERE}/runs/gsm8k_counterfactual_${RUN_STAMP}}

test -f "${TABLE}"
test ! -e "${OUT_DIR}"
mkdir -p "${OUT_DIR}/trace_run"
export PYTHONPATH="${REPO_ROOT}/python:${LIGHTEVAL_SRC}:${HERE}:${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

cat >"${OUT_DIR}/config.env" <<EOF
MODEL=${MODEL}
TABLE=${TABLE}
TASK=gsm8k|0
GSM8K_LIMIT=${GSM8K_LIMIT}
TRACE_GENERATION=${GEN_KWARGS}
TREE_SHAPE=steps3_topk1_draft4
COUNTERFACTUAL_EVENT_FILTER=${COUNTERFACTUAL_EVENT_FILTER}
COUNTERFACTUAL_MAX_EVENTS=${COUNTERFACTUAL_MAX_EVENTS}
COUNTERFACTUAL_ONE_EVENT_PER_REQUEST=1
COUNTERFACTUAL_TEMPERATURE=${COUNTERFACTUAL_TEMPERATURE}
COUNTERFACTUAL_MAX_NEW_TOKENS=${COUNTERFACTUAL_MAX_NEW_TOKENS}
EOF

echo "[$(date -Is)] full GSM8K trace start"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${RUNNER}" \
  --model "${MODEL}" --tokenizer "${MODEL}" --tasks 'gsm8k|0' \
  --limit "${GSM8K_LIMIT}" \
  --output-dir "${OUT_DIR}/trace_run/lighteval_tracker" \
  --output-path "${OUT_DIR}/trace_run/lighteval_results.json" \
  --metrics-output-path "${OUT_DIR}/trace_run/sglang_metrics.json" \
  --run-tag "qwen35b_mtp314_gsm8k_full_rejection_trace" \
  --mode csd --server-log "${OUT_DIR}/trace_run/server.log" \
  --cuda-devices "${CUDA_DEVICES}" --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --skip-server-warmup --watchdog-timeout 7200 \
  --mamba-scheduler-strategy no_buffer \
  --max-gen-toks "${GSM8K_MAX_GEN_TOKS}" --max-length "${MAX_LENGTH}" \
  --trust-remote-code --override-chat-template auto --enable-thinking true \
  --gen-kwargs "${GEN_KWARGS}" --save-details --generation-only \
  --force-num-samples 1 --disable-sample-cache --dataset-loading-processes 1 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --csd-enabled --csd-table-path "${TABLE}" --csd-freq-threshold 6 \
  --csd-prob-ratio 0.3 --csd-force-accept-disabled \
  --csd-rejection-trace --csd-rejection-trace-dir "${OUT_DIR}/trace_run/trace" \
  --csd-rejection-trace-capacity "${TRACE_CAPACITY}" \
  2>&1 | tee "${OUT_DIR}/trace_run/run.log"

echo "[$(date -Is)] paired counterfactual replay start"
"${PYTHON}" "${HERE}/replay_gsm8k_counterfactual.py" \
  --trace-dir "${OUT_DIR}/trace_run/trace" \
  --output "${OUT_DIR}/counterfactual_events.jsonl" \
  --summary "${OUT_DIR}/counterfactual_summary.json" \
  --model "${MODEL}" --tp-size "${TP_SIZE}" --cuda-devices "${CUDA_DEVICES}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" --context-length "${MAX_LENGTH}" \
  --max-new-tokens "${COUNTERFACTUAL_MAX_NEW_TOKENS}" \
  --batch-size "${COUNTERFACTUAL_BATCH_SIZE}" \
  --max-running-requests "${COUNTERFACTUAL_MAX_RUNNING_REQUESTS}" \
  --temperature "${COUNTERFACTUAL_TEMPERATURE}" --top-p 0.95 --top-k 20 \
  --sample-seed 42 --seed 1234 --max-events "${COUNTERFACTUAL_MAX_EVENTS}" \
  --event-filter "${COUNTERFACTUAL_EVENT_FILTER}" --one-event-per-request \
  2>&1 | tee "${OUT_DIR}/counterfactual_replay.log"

echo "[$(date -Is)] complete: ${OUT_DIR}"
