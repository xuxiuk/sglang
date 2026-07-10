#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

# ==================== 基础配置 ====================
BASE_SCRIPT=${BASE_SCRIPT:-"${SCRIPT_DIR}/run_lighteval_csd_decoding_comparison_dyn.sh"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_decoding_comparison_sweep/${RUN_STAMP}"}
RESULT_FILE=${RESULT_FILE:-"${OUT_DIR}/results/csd_decoding_comparison_sweep.jsonl"}

# ==================== 模型路径控制 (新加) ====================
MODEL_PATH=${MODEL_PATH:-"/root/model/Qwen3.6-35B-A3B"}

# ==================== 硬件与网络控制 ====================
CUDA_DEVICES=${CUDA_DEVICES:-"4,5,6,7"}
TP_SIZE=${TP_SIZE:-4}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.9}
PORT=${PORT:-30002}

# ==================== 扫描参数配置 ====================
CSD_PROB_RATIOS=${CSD_PROB_RATIOS:-"0.01 0.1 0.3"}
# 这里将原本的 SPEC_TOPKS 改为扫描 CSD_REBUILD_TOP_KEEPS (即 rebuild topk)
CSD_REBUILD_TOP_KEEPS=${CSD_REBUILD_TOP_KEEPS:-"2000 5000 10000 15000 20000"}

# 投机采样基础算法参数 (保持固定默认值，不与 rebuild topk 混淆)
SPEC_TOPK=${SPEC_TOPK:-1}
SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-5}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-5}

# 错误控制
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}

mkdir -p "${OUT_DIR}/results"

# ==================== 核心执行函数 ====================
run_one() {
  local method="$1"
  local prob_ratio="$2"
  local top_keep="$3"

  echo "------------------------------------------------------------"
  echo "Running sweep item:"
  echo "  METHOD           : ${method}"
  echo "  MODEL_PATH       : ${MODEL_PATH}"
  echo "  CSD_PROB_RATIO   : ${prob_ratio}"
  echo "  CSD_REBUILD_TOP  : ${top_keep:-None}"
  echo "  CUDA_DEVICES     : ${CUDA_DEVICES}"
  echo "  TP_SIZE          : ${TP_SIZE}"
  echo "  MEM_FRACTION     : ${MEM_FRACTION_STATIC}"
  echo "  PORT             : ${PORT}"
  echo "------------------------------------------------------------"

  # 导出变量并调用基础脚本
  # 增加了 MODEL_PATH 和 TOKENIZER_PATH 的透传
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  PORT="${PORT}" \
  TP_SIZE="${TP_SIZE}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  MODEL_PATH="${MODEL_PATH}" \
  TOKENIZER_PATH="${MODEL_PATH}" \
  RUN_STAMP="${RUN_STAMP}" \
  OUT_DIR="${OUT_DIR}" \
  RESULT_FILE="${RESULT_FILE}" \
  METHOD_FILTER="${method}" \
  CSD_PROB_RATIO="${prob_ratio}" \
  CSD_REBUILD_TOP_KEEP="${top_keep}" \
  SPEC_TOPK="${SPEC_TOPK}" \
  SPEC_NUM_STEPS="${SPEC_NUM_STEPS}" \
  SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS}" \
    bash "${BASE_SCRIPT}"
}

status=0
run_status=0

# ============================================================
# 1. 首先跑 基线方法 (Baseline & Vanilla)
# ============================================================
# 这里会顺次跑完完全不带投机的 baseline 以及带投机不带 CSD 的 vanilla
for method in baseline vanilla; do
  echo ">>> [Phase 1/3] Running Base Method: ${method}"
  run_one "${method}" "0.0" "" || run_status=$?
  if [[ "${run_status}" != "0" ]]; then
    status="${run_status}"
    echo "Base run failed: method=${method}, status=${run_status}" >&2
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then exit "${run_status}"; fi
  fi
done

# ============================================================
# 2. 跑静态 CSD Sweep (只扫描 prob_ratio，不需要 rebuild_top_keep)
# ============================================================
echo ">>> [Phase 2/3] Starting Static CSD Sweep..."
for prob_ratio in ${CSD_PROB_RATIOS}; do
  for method in csd_plain_table_static; do
    run_status=0
    run_one "${method}" "${prob_ratio}" "" || run_status=$?
    if [[ "${run_status}" != "0" ]]; then
      status="${run_status}"
      echo "Static sweep failed: method=${method}, ratio=${prob_ratio}, status=${run_status}" >&2
      if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then exit "${run_status}"; fi
    fi
  done
done

# ============================================================
# 3. 跑动态 CSD Sweep (同时扫描 prob_ratio 和 rebuild_top_keep)
# ============================================================
echo ">>> [Phase 3/3] Starting Dynamic CSD Sweep..."
for prob_ratio in ${CSD_PROB_RATIOS}; do
  for top_keep in ${CSD_REBUILD_TOP_KEEPS}; do
    for method in csd_plain_table_top5; do
      run_status=0
      run_one "${method}" "${prob_ratio}" "${top_keep}" || run_status=$?
      if [[ "${run_status}" != "0" ]]; then
        status="${run_status}"
        echo "Dynamic sweep failed: method=${method}, ratio=${prob_ratio}, top_keep=${top_keep}, status=${run_status}" >&2
        if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then exit "${run_status}"; fi
      fi
    done
  done
done

# ==================== 结束汇报 ====================
echo "================================================------------"
echo "All Sweep Phases Done."
echo "OUT_DIR=${OUT_DIR}"
echo "RESULT_FILE=${RESULT_FILE}"
exit "${status}"