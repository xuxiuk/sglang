#!/usr/bin/env bash
# Run the 515 LightEval CSD matrix with the 515/max_new_tokens=1024 RedPajama table
# on two independent 4-GPU groups. The method matrix is split across groups and
# merged into one JSONL at the end.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script; run: bash ${BASH_SOURCE[0]}" >&2
  return 0
fi
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="/root/miniconda3/envs/sglang/bin:${CUDA_HOME}/bin:/home/ccuser/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Use local SGLang source and the installed sgl-kernel wheel, matching the prior run.
export SGLANG_SRC=${SGLANG_SRC:-"${REPO_ROOT}/python"}
export LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-"${REPO_ROOT}/benchmark/csd/lighteval/src"}
export PYTHONPATH="${SGLANG_SRC}:${LIGHTEVAL_SRC}:${REPO_ROOT}:${PYTHONPATH:-}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_515_calib1024_two_gpu_groups}
OUT_ROOT_BASE=${OUT_ROOT_BASE:-"${REPO_ROOT}/benchmark/csd/runs/lighteval_csd_classic_tree_shape_sweep/${RUN_STAMP}"}
MERGED_RESULT_FILE=${MERGED_RESULT_FILE:-"${OUT_ROOT_BASE}/results/classic_tree_shape_sweep.jsonl"}
DRIVER_LOG_DIR=${DRIVER_LOG_DIR:-"${OUT_ROOT_BASE}/driver_logs"}

PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/runs/redpajama/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json"}

TREE_SHAPES=${TREE_SHAPES:-"515:5:1:5"}
TASK_FILTER=${TASK_FILTER:-all}
# Keep the previous 515 matrix and run the entropy thresholds for both
# ratio-gate modes. p20 is looser (higher threshold), p40 is stricter.
METHOD_SET_GROUP_A=${METHOD_SET_GROUP_A:-"auto eagle plain dynamic_entropy_p20 dynamic_entropy dynamic_entropy_p40"}
METHOD_SET_GROUP_B=${METHOD_SET_GROUP_B:-"dynamic dynamic_ignore_ratio dynamic_entropy_p20_ignore_ratio dynamic_entropy_ignore_ratio dynamic_entropy_p40_ignore_ratio"}

GPU_GROUP_A=${GPU_GROUP_A:-0,1,2,3}
GPU_GROUP_B=${GPU_GROUP_B:-4,5,6,7}
PORT_GROUP_A=${PORT_GROUP_A:-30011}
PORT_GROUP_B=${PORT_GROUP_B:-30012}
TP_SIZE_GROUP_A=${TP_SIZE_GROUP_A:-4}
TP_SIZE_GROUP_B=${TP_SIZE_GROUP_B:-4}

MODEL_PATH=${MODEL_PATH:-/root/model/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-7200}
MAMBA_SCHEDULER_STRATEGY=${MAMBA_SCHEDULER_STRATEGY:-no_buffer}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}
SHOW_KERNEL=${SHOW_KERNEL:-1}
USE_SOURCE_SGL_KERNEL=${USE_SOURCE_SGL_KERNEL:-0}
COPY_INCREMENTAL_KERNEL=${COPY_INCREMENTAL_KERNEL:-0}

# CSD/eval settings matching the previous 313/515 run, except for the new 515 table.
CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-6}
CSD_KEY_SELECTION_STRATEGY=${CSD_KEY_SELECTION_STRATEGY:-frequency}
CSD_SCORE_THRESHOLD=${CSD_SCORE_THRESHOLD:-0}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
CSD_TABLE_PROB_RATIO=${CSD_TABLE_PROB_RATIO:-0.3}
PLAIN_CSD_TABLE_PROB_RATIO=${PLAIN_CSD_TABLE_PROB_RATIO:-0.01}
CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO=${CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO:-0}
ENTROPY_P20_THRESHOLD=${ENTROPY_P20_THRESHOLD:-1.5638477802276611}
ENTROPY_P30_THRESHOLD=${ENTROPY_P30_THRESHOLD:-1.3415851593017578}
ENTROPY_P40_THRESHOLD=${ENTROPY_P40_THRESHOLD:-1.1563854217529297}

if [[ ! -f "${PLAIN_CSD_TABLE_PATH}" ]]; then
  echo "CSD table not found: ${PLAIN_CSD_TABLE_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT_BASE}/results" "${DRIVER_LOG_DIR}"

run_group() {
  local group_name="$1"
  local cuda_devices="$2"
  local port="$3"
  local tp_size="$4"
  local method_set="$5"
  local out_root="${OUT_ROOT_BASE}/${group_name}"

  echo "[${group_name}] CUDA_DEVICES=${cuda_devices} PORT=${port} TP_SIZE=${tp_size} METHOD_SET=${method_set}"

  env \
    RUN_STAMP="${RUN_STAMP}" \
    OUT_ROOT="${out_root}" \
    RESULT_FILE="${out_root}/results/classic_tree_shape_sweep.jsonl" \
    MODEL_PATH="${MODEL_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    CUDA_DEVICES="${cuda_devices}" \
    TP_SIZE="${tp_size}" \
    PORT="${port}" \
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
    WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT}" \
    MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY}" \
    CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE}" \
    SHOW_KERNEL="${SHOW_KERNEL}" \
    USE_SOURCE_SGL_KERNEL="${USE_SOURCE_SGL_KERNEL}" \
    COPY_INCREMENTAL_KERNEL="${COPY_INCREMENTAL_KERNEL}" \
    TREE_SHAPES="${TREE_SHAPES}" \
    TASK_FILTER="${TASK_FILTER}" \
    METHOD_SET="${method_set}" \
    PLAIN_CSD_TABLE_PATH="${PLAIN_CSD_TABLE_PATH}" \
    CSD_FREQ_THRESHOLD="${CSD_FREQ_THRESHOLD}" \
    CSD_KEY_SELECTION_STRATEGY="${CSD_KEY_SELECTION_STRATEGY}" \
    CSD_SCORE_THRESHOLD="${CSD_SCORE_THRESHOLD}" \
    CSD_PROB_RATIO="${CSD_PROB_RATIO}" \
    CSD_TABLE_PROB_RATIO="${CSD_TABLE_PROB_RATIO}" \
    PLAIN_CSD_TABLE_PROB_RATIO="${PLAIN_CSD_TABLE_PROB_RATIO}" \
    CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO="${CSD_DYNAMIC_UPDATE_IGNORE_PROB_RATIO}" \
    ENTROPY_P20_THRESHOLD="${ENTROPY_P20_THRESHOLD}" \
    ENTROPY_P30_THRESHOLD="${ENTROPY_P30_THRESHOLD}" \
    ENTROPY_P40_THRESHOLD="${ENTROPY_P40_THRESHOLD}" \
    bash benchmark/csd/eval/run_lighteval_csd_classic_tree_shape_sweep.sh
}

status=0
run_group group_a "${GPU_GROUP_A}" "${PORT_GROUP_A}" "${TP_SIZE_GROUP_A}" "${METHOD_SET_GROUP_A}" \
  >"${DRIVER_LOG_DIR}/group_a.log" 2>&1 &
pid_a=$!
run_group group_b "${GPU_GROUP_B}" "${PORT_GROUP_B}" "${TP_SIZE_GROUP_B}" "${METHOD_SET_GROUP_B}" \
  >"${DRIVER_LOG_DIR}/group_b.log" 2>&1 &
pid_b=$!

echo "Started group_a pid=${pid_a}, log=${DRIVER_LOG_DIR}/group_a.log"
echo "Started group_b pid=${pid_b}, log=${DRIVER_LOG_DIR}/group_b.log"

wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?

: >"${MERGED_RESULT_FILE}"
for group_name in group_a group_b; do
  group_result="${OUT_ROOT_BASE}/${group_name}/results/classic_tree_shape_sweep.jsonl"
  if [[ -f "${group_result}" ]]; then
    cat "${group_result}" >>"${MERGED_RESULT_FILE}"
  else
    echo "Missing group result: ${group_result}" >&2
    status=1
  fi
done

python - "${MERGED_RESULT_FILE}" <<'PY'
import json
import sys
from collections import Counter
path = sys.argv[1]
rows = []
try:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
except FileNotFoundError:
    pass
print({"merged_result": path, "rows": len(rows)})
methods = []
for row in rows:
    tag = row.get("other", {}).get("run_tag", "")
    if tag.startswith("dynamic_entropy_p20_ignore_ratio_"):
        method = "dynamic_entropy_p20_ignore_ratio"
    elif tag.startswith("dynamic_entropy_p40_ignore_ratio_"):
        method = "dynamic_entropy_p40_ignore_ratio"
    elif tag.startswith("dynamic_entropy_ignore_ratio_"):
        method = "dynamic_entropy_ignore_ratio"
    elif tag.startswith("dynamic_ignore_ratio_"):
        method = "dynamic_ignore_ratio"
    elif tag.startswith("dynamic_entropy_p20_"):
        method = "dynamic_entropy_p20"
    elif tag.startswith("dynamic_entropy_p40_"):
        method = "dynamic_entropy_p40"
    elif tag.startswith("dynamic_entropy_"):
        method = "dynamic_entropy"
    else:
        method = tag.split("_", 1)[0] if tag else "unknown"
    methods.append(method)
print("methods", dict(Counter(methods)))
PY

echo "Done. OUT_ROOT_BASE=${OUT_ROOT_BASE}"
echo "MERGED_RESULT_FILE=${MERGED_RESULT_FILE}"
exit "${status}"
