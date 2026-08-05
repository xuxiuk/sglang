#!/usr/bin/env bash
# Formal Qwen3.5-35B-A3B quality matrix on two independent 4-GPU groups.
# Protocol: AIME25 avg@16, Math500 avg@4, LiveCodeBench v6 pass@1 from 4 samples.
set -euo pipefail

RUN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/root/sglang-dspark-csd
PYTHON=/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python
MODEL=/data/model/Qwen3.5-35B-A3B
TABLE=${RUN_DIR}/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps3_topk1_draft3_temp1.0_ratio0.01.json
MATRIX=${RUN_DIR}/vendor/eval/libexec/lighteval_matrix.bash
METHODS_ALL="auto eagle plain dynamic dynamic_entropy_p20_ignore_ratio"
METHODS_SPEC="eagle plain dynamic dynamic_entropy_p20_ignore_ratio"
GEN_KWARGS="temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0,seed=1234"

export NCCL_IB_DISABLE=1

run_matrix() {
  local gpu_devices=$1
  local port=$2
  local group=$3
  local task_filter=$4
  local methods=$5
  local limit=${6:-}
  local max_gen=${7:-81920}
  local max_length=${8:-96000}

  env \
    REPO_ROOT_OVERRIDE="${REPO_ROOT}" \
    LIGHTEVAL_RUNNER="${RUN_DIR}/vendor/eval/run_lighteval_sglang_native.py" \
    LIGHTEVAL_SRC="${RUN_DIR}/vendor/lighteval/src" \
    SGLANG_SRC="${REPO_ROOT}/python" \
    LIGHTEVAL_PYTHON="${PYTHON}" \
    MODEL_PATH="${MODEL}" TOKENIZER_PATH="${MODEL}" \
    CUDA_DEVICES="${gpu_devices}" PORT="${port}" TP_SIZE=4 \
    TREE_SHAPES=314:3:1:4 METHOD_SET="${methods}" \
    TASK_FILTER="${task_filter}" LIMIT="${limit}" \
    LCB_TASK=lcb:codegeneration_v6 AIME_TASK=aime25_avg MATH500_TASK=math_500 \
    GSM8K_TASK="${GSM8K_TASK:-gsm8k}" \
    MAX_RUNNING_REQUESTS=48 \
    PLAIN_CSD_TABLE_PATH="${TABLE}" \
    CSD_FREQ_THRESHOLD=6 CSD_PROB_RATIO=0.3 \
    CSD_REBUILD_THRESHOLD=4096 \
    ENTROPY_P20_THRESHOLD=1.5638477802276611 \
    CODING_RECOMMENDED_GEN_KWARGS="${GEN_KWARGS}" \
    THINK_GENERAL_RECOMMENDED_GEN_KWARGS="${GEN_KWARGS}" \
    LCB_RECOMMENDED_GEN_KWARGS="${GEN_KWARGS}" \
    LCB_MAX_GEN_TOKS="${max_gen}" LCB_MAX_LENGTH="${max_length}" \
    AIME_MAX_GEN_TOKS="${max_gen}" AIME_MAX_LENGTH="${max_length}" \
    MATH500_MAX_GEN_TOKS="${max_gen}" MATH500_MAX_LENGTH="${max_length}" \
    RUN_STAMP="avg314_c48_20260802_${group}" \
    OUT_ROOT="${RUN_DIR}/${group}" \
    CONTINUE_ON_FAILURE=0 SHOW_KERNEL=1 \
    bash "${MATRIX}"
}

run_group0_queue() {
  # GPU 0-3: LCB full matrix, Math500 Auto, then GSM8K full matrix.
  run_matrix 0,1,2,3 32120 gpu0_3 lcb "${METHODS_ALL}"
  run_matrix 0,1,2,3 32120 gpu0_3 math500 auto
  GSM8K_TASK=gsm8k_avg \
    run_matrix 0,1,2,3 32120 gsm8k_avg_gpu0_3 gsm8k "${METHODS_ALL}"
}

run_group1_queue() {
  # GPU 4-7: AIME full matrix, then speculative Math500 methods.
  run_matrix 4,5,6,7 32140 gpu4_7 aime25 "${METHODS_ALL}"
  run_matrix 4,5,6,7 32140 gpu4_7 math500 "${METHODS_SPEC}"
}

case "${1:-}" in
  all)
    # One formal entry point. Each four-GPU queue is sequential internally;
    # the two disjoint GPU queues run concurrently and use distinct ports.
    run_group0_queue &
    pid_group0=$!
    run_group1_queue &
    pid_group1=$!
    status=0
    wait "$pid_group0" || status=$?
    wait "$pid_group1" || status=$?
    exit "$status"
    ;;
  smoke0)
    run_matrix 0,1,2,3 32120 smoke_gpu0_3 aime25 auto 1 128 4096
    ;;
  smoke1)
    run_matrix 4,5,6,7 32140 smoke_gpu4_7 math500 eagle 1 128 4096
    ;;
  smoke_lcb)
    run_matrix 0,1,2,3 32120 smoke_lcb_dual_metric lcb auto 1 128 4096
    ;;
  group0)
    run_group0_queue
    ;;
  group1)
    run_group1_queue
    ;;
  math500_presence0_gpu4_7)
    # Isolated Math500 rerun: change only presence_penalty from 1.5 to 0.0.
    # Keep all artifacts separate from the formal matrix above.
    GEN_KWARGS="temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0,seed=1234"
    run_matrix 4,5,6,7 32140 math500_presence0_gpu4_7 math500 "${METHODS_ALL}"
    ;;
  lcb_entropy_p30_gpu0_3)
    # Isolated LCB entropy-gate comparison: p30 threshold, otherwise identical
    # to the formal p20 dynamic+entropy configuration.
    run_matrix 0,1,2,3 32120 lcb_entropy_p30_gpu0_3 lcb "dynamic_entropy_ignore_ratio"
    ;;
  gsm8k_avg_gpu0_3)
    # Recovery-only entry for rerunning GSM8K without the complete formal queue.
    GSM8K_TASK=gsm8k_avg
    run_matrix 0,1,2,3 32120 gsm8k_avg_gpu0_3 gsm8k "${METHODS_ALL}"
    ;;
  math500_entropy_p30_gpu0_3)
    # Isolated Math500 p30 entropy-gate comparison; all other formal
    # generation and CSD parameters remain unchanged.
    run_matrix 0,1,2,3 32120 math500_entropy_p30_gpu0_3 math500 "dynamic_entropy_ignore_ratio"
    ;;
  *)
    echo "Usage: $0 {all|smoke0|smoke1|smoke_lcb|group0|group1|math500_presence0_gpu4_7|lcb_entropy_p30_gpu0_3|gsm8k_avg_gpu0_3|math500_entropy_p30_gpu0_3}" >&2
    exit 2
    ;;
esac
