#!/usr/bin/env bash
# APPS/TACO code-OOD generation: four samples per question. The companion
# evaluator computes avg@4 and pass@4 with the official TACO/APPS test runner.
set -euo pipefail

RUN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/root/sglang-dspark-csd
PYTHON=/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python
MODEL=/data/model/Qwen3.5-35B-A3B
TABLE=${RUN_DIR}/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps3_topk1_draft3_temp1.0_ratio0.01.json
MATRIX=${RUN_DIR}/vendor/eval/libexec/code_ood_matrix.bash
METHODS=${METHODS_OVERRIDE:-"eagle plain dynamic dynamic_entropy_p20_ignore_ratio"}

run_dataset() {
  local task=$1
  local devices=$2
  local port=$3
  local nccl_port=$4
  local data_file=$5

  env \
    REPO_ROOT_OVERRIDE="${REPO_ROOT}" \
    EVAL_ROOT="${RUN_DIR}/vendor/eval" \
    PYTHON="${PYTHON}" MODEL_PATH="${MODEL}" TOKENIZER_PATH="${MODEL}" \
    CUDA_DEVICES="${devices}" PORT="${port}" NCCL_PORT="${nccl_port}" TP_SIZE=4 \
    TREE_LABEL=314 SPEC_NUM_STEPS=3 SPEC_TOPK=1 SPEC_DRAFT_TOKENS=4 \
    METHOD_SET="${METHODS}" MAX_RUNNING_REQUESTS=48 PARALLEL=48 \
    OOD_DATA_FILES="${data_file}" NUM_EXAMPLES=1000 NUM_SAMPLES=4 \
    MAX_NEW_TOKENS=81920 \
    TEMPERATURE=1.0 TOP_P=0.95 TOP_K=20 MIN_P=0.0 \
    PRESENCE_PENALTY=1.5 REPETITION_PENALTY=1.0 ENABLE_THINKING=true \
    PLAIN_CSD_TABLE_PATH="${TABLE}" \
    CSD_FREQ_THRESHOLD=6 CSD_PROB_RATIO=0.3 \
    CSD_REBUILD_THRESHOLD=4096 \
    ENTROPY_P20_THRESHOLD=1.5638477802276611 \
    RUN_STAMP="${task}_n4_314_c48_20260802" \
    OUT_ROOT="${RUN_DIR}/code_ood_n4/${task}_queue" \
    bash "${MATRIX}"
}

case "${1:-}" in
  apps)
    run_dataset apps 0,1,2,3 32220 32221 \
      "${RUN_DIR}/vendor/assets/data/apps_2000_code_ood_alpaca.json"
    ;;
  taco)
    run_dataset taco 0,1,2,3 32240 32241 \
      "${RUN_DIR}/vendor/assets/data/taco_1000_code_ood_alpaca.json"
    ;;
  all_gpu0_3)
    run_dataset apps 0,1,2,3 32220 32221 \
      "${RUN_DIR}/vendor/assets/data/apps_2000_code_ood_alpaca.json"
    run_dataset taco 0,1,2,3 32240 32241 \
      "${RUN_DIR}/vendor/assets/data/taco_1000_code_ood_alpaca.json"
    ;;
  *)
    echo "Usage: $0 {apps|taco|all_gpu0_3}" >&2
    exit 2
    ;;
esac
