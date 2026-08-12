#!/usr/bin/env bash
# Qwen3.5-397B-A17B-FP8 formal MTP+CSD evaluation on one 8xH20 group.
# Four independent n=1 generations per problem; report avg@4 and pass@4.
set -euo pipefail

RUN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/root/sglang-dspark-csd
PYTHON=/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python
MODEL=/data/model/Qwen3.5-397B-A17B-FP8
TABLE=${RUN_DIR}/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-397B-A17B-FP8_mtp_EAGLE_steps3_topk1_draft3_temp1.0_ratio0.01.json
EVAL_BUNDLE=/root/sglang-dspark-csd/runs/mtp_csd/qwen35b_mtp314_final/vendor
MATRIX=$EVAL_BUNDLE/eval/libexec/lighteval_matrix.bash
RUNNER=$EVAL_BUNDLE/eval/run_lighteval_sglang_native.py
LIGHTEVAL_SRC=$EVAL_BUNDLE/lighteval/src

# Match the four DSpark comparison methods. In the MTP matrix, `eagle` is bare MTP.
METHODS="eagle plain dynamic dynamic_entropy_p20_ignore_ratio"
GEN_KWARGS="temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0,seed=1234"

export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

run_task() {
  local task_filter=$1
  local task_name=$2
  local port=$3
  local limit=${4:-}
  # Math500 uses the accuracy-preserving ratio selected by the 0.2--0.5
  # sweep. Keep the established 0.3 setting for the other formal tasks.
  local csd_prob_ratio=0.3
  if [[ "${task_filter}" == "math500" ]]; then
    csd_prob_ratio=0.4
  fi

  env \
    REPO_ROOT_OVERRIDE="${REPO_ROOT}" \
    LIGHTEVAL_RUNNER="${RUNNER}" \
    LIGHTEVAL_SRC="${LIGHTEVAL_SRC}" \
    SGLANG_SRC="${REPO_ROOT}/python" \
    LIGHTEVAL_PYTHON="${PYTHON}" \
    MODEL_PATH="${MODEL}" TOKENIZER_PATH="${MODEL}" \
    CUDA_DEVICES=0,1,2,3,4,5,6,7 PORT="${port}" TP_SIZE=8 \
    MEM_FRACTION_STATIC=0.85 \
    TREE_SHAPES=314:3:1:4 METHOD_SET="${METHODS}" \
    TASK_FILTER="${task_filter}" LIMIT="${limit}" \
    LCB_TASK=lcb:codegeneration_v6 AIME_TASK=aime25_avg MATH500_TASK=math_500 \
    GSM8K_TASK=gsm8k_avg \
    MAX_RUNNING_REQUESTS=48 \
    PLAIN_CSD_TABLE_PATH="${TABLE}" \
    CSD_FREQ_THRESHOLD=6 CSD_PROB_RATIO="${csd_prob_ratio}" \
    CSD_REBUILD_THRESHOLD=4096 \
    ENTROPY_P20_THRESHOLD=1.5638477802276611 \
    CODING_RECOMMENDED_GEN_KWARGS="${GEN_KWARGS}" \
    THINK_GENERAL_RECOMMENDED_GEN_KWARGS="${GEN_KWARGS}" \
    LCB_RECOMMENDED_GEN_KWARGS="${GEN_KWARGS}" \
    LCB_MAX_GEN_TOKS=81920 LCB_MAX_LENGTH=96000 \
    AIME_MAX_GEN_TOKS=81920 AIME_MAX_LENGTH=96000 \
    MATH500_MAX_GEN_TOKS=81920 MATH500_MAX_LENGTH=96000 \
    RUN_STAMP="qwen397b_avg314_tp8_c48_20260808_${task_name}" \
    OUT_ROOT="${RUN_DIR}/results/${task_name}" \
    CONTINUE_ON_FAILURE=0 SHOW_KERNEL=1 \
    bash "${MATRIX}"
}

case "${1:-all}" in
  all)
    run_task lcb lcb_avg4 40100
    run_task aime25 aime25_avg4 40132
    run_task math500 math500_avg4 40164
    run_task gsm8k gsm8k_avg4 40196
    ;;
  smoke)
    METHODS=eagle run_task math500 smoke_math500_bare 40100 1
    ;;
  lcb|aime25|math500|gsm8k)
    case "$1" in
      lcb) run_task lcb lcb_avg4 40100 ;;
      aime25) run_task aime25 aime25_avg4 40132 ;;
      math500) run_task math500 math500_avg4 40164 ;;
      gsm8k) run_task gsm8k gsm8k_avg4 40196 ;;
    esac
    ;;
  *)
    echo "Usage: $0 {all|smoke|lcb|aime25|math500|gsm8k}" >&2
    exit 2
    ;;
esac
