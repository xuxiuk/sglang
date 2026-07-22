#!/usr/bin/env bash
# Internal APPS/TACO matrix used by run_mtp515_v0510_six_datasets_mr48.sh.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
cd "${REPO_ROOT}"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_domain_ood_csd_515}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/benchmark/csd/runs/domain_ood_csd_515/${RUN_STAMP}"}
mkdir -p "${OUT_ROOT}"

PYTHON=${PYTHON:-/root/miniconda3/envs/sglang/bin/python}
CONDA_BIN=$(dirname "${PYTHON}")
MODEL_PATH=${MODEL_PATH:-/root/model/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
MODEL_ID=${MODEL_ID:-Qwen3.5-35B-A3B}
CUDA_DEVICES=${CUDA_DEVICES:-4,5,6,7}
TP_SIZE=${TP_SIZE:-4}
PORT=${PORT:-30072}
HOST=${HOST:-127.0.0.1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.72}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-24}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-7200}
MAMBA_SCHEDULER_STRATEGY=${MAMBA_SCHEDULER_STRATEGY:-no_buffer}
PARALLEL=${PARALLEL:-24}
ENABLE_SERVER_METRICS=${ENABLE_SERVER_METRICS:-true}
DECODE_LOG_INTERVAL=${DECODE_LOG_INTERVAL:-40}
METRICS_SCRAPE_INTERVAL=${METRICS_SCRAPE_INTERVAL:-30}

OOD_DATA_FILES=${OOD_DATA_FILES:-}
NUM_EXAMPLES=${NUM_EXAMPLES:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-1.5}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-}
ENABLE_THINKING=${ENABLE_THINKING:-true}

TREE_LABEL=${TREE_LABEL:-515}
SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-5}
SPEC_TOPK=${SPEC_TOPK:-1}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-5}
METHOD_SET=${METHOD_SET:-"eagle plain dynamic_ignore_ratio dynamic_entropy_p20_ignore_ratio"}

CSD_FREQ_THRESHOLD=${CSD_FREQ_THRESHOLD:-6}
CSD_KEY_SELECTION_STRATEGY=${CSD_KEY_SELECTION_STRATEGY:-frequency}
CSD_SCORE_THRESHOLD=${CSD_SCORE_THRESHOLD:-0}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD:-512}
CSD_HUGE_REBUILD_THRESHOLD=${CSD_HUGE_REBUILD_THRESHOLD:-1000000000}
ENTROPY_P20_THRESHOLD=${ENTROPY_P20_THRESHOLD:-1.5638477802276611}
PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/assets/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json"}

if [[ -z "${OOD_DATA_FILES}" ]]; then
  echo "OOD_DATA_FILES must be set to one or more prepared JSON/JSONL files." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PATH="${CONDA_BIN}:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}:${PYTHONPATH:-}"
export NO_PROXY="127.0.0.1,localhost,${HOST},${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${HOST},${no_proxy:-}"

safe_name() {
  local value="$1"
  value="${value//[^A-Za-z0-9._-]/-}"
  value="${value##[-_]}"
  value="${value%%[-_]}"
  [[ -n "${value}" ]] || value=none
  printf '%s' "${value}"
}

server_args_for_method() {
  local method="$1"
  local args=(
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --tp-size "${TP_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --watchdog-timeout "${WATCHDOG_TIMEOUT}"
    --mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY}"
    --trust-remote-code
    --disable-radix-cache
    --disable-overlap-schedule
    --log-level warning
  )
  if [[ "${ENABLE_SERVER_METRICS}" == "true" ]]; then
    args+=(
      --enable-metrics
      --decode-log-interval "${DECODE_LOG_INTERVAL}"
    )
  fi

  if [[ "${method}" != "auto" ]]; then
    args+=(
      --speculative-algorithm EAGLE
      --speculative-num-steps "${SPEC_NUM_STEPS}"
      --speculative-eagle-topk "${SPEC_TOPK}"
      --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}"
    )
  fi

  case "${method}" in
    auto|eagle|vanilla)
      ;;
    plain)
      args+=(
        --speculative-csd
        --speculative-csd-table-path "${PLAIN_CSD_TABLE_PATH}"
        --speculative-csd-freq-threshold "${CSD_FREQ_THRESHOLD}"
        --speculative-csd-key-selection-strategy "${CSD_KEY_SELECTION_STRATEGY}"
        --speculative-csd-score-threshold "${CSD_SCORE_THRESHOLD}"
        --speculative-csd-prob-ratio "${CSD_PROB_RATIO}"
      )
      ;;
    dynamic|dynamic_ignore_ratio|dynamic_entropy_p20_ignore_ratio|dynamic_ignore_ratio_huge|dynamic_entropy_p20_ignore_ratio_huge)
      local rebuild_threshold="${CSD_REBUILD_THRESHOLD}"
      if [[ "${method}" == *_huge ]]; then
        rebuild_threshold="${CSD_HUGE_REBUILD_THRESHOLD}"
      fi
      args+=(
        --speculative-csd
        --speculative-csd-table-path "${PLAIN_CSD_TABLE_PATH}"
        --speculative-csd-freq-threshold "${CSD_FREQ_THRESHOLD}"
        --speculative-csd-key-selection-strategy "${CSD_KEY_SELECTION_STRATEGY}"
        --speculative-csd-score-threshold "${CSD_SCORE_THRESHOLD}"
        --speculative-csd-prob-ratio "${CSD_PROB_RATIO}"
        --speculative-csd-dynamic-update
        --speculative-csd-rebuild-threshold "${rebuild_threshold}"
      )
      if [[ "${method}" != "dynamic" ]]; then
        args+=(--speculative-csd-dynamic-update-ignore-prob-ratio)
      fi
      if [[ "${method}" == dynamic_entropy_p20_ignore_ratio* ]]; then
        args+=(--speculative-csd-force-accept-entropy-threshold "${ENTROPY_P20_THRESHOLD}")
      fi
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 1
      ;;
  esac

  printf '%q ' "${args[@]}"
}

wait_server() {
  local deadline=$((SECONDS + 900))
  until curl --noproxy '*' -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "Server did not become ready before timeout." >&2
      return 1
    fi
    sleep 2
  done
}

kill_server() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 5
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi
}

if [[ ! -f "${PLAIN_CSD_TABLE_PATH}" ]]; then
  echo "CSD table not found: ${PLAIN_CSD_TABLE_PATH}" >&2
  exit 1
fi

cat <<EOF | tee "${OUT_ROOT}/apps_taco_config.txt"
OUT_ROOT=${OUT_ROOT}
OOD_DATA_FILES=${OOD_DATA_FILES}
NUM_EXAMPLES=${NUM_EXAMPLES:-all}
MODEL_PATH=${MODEL_PATH}
CUDA_DEVICES=${CUDA_DEVICES}
TP_SIZE=${TP_SIZE}
PORT=${PORT}
METHOD_SET=${METHOD_SET}
PARALLEL=${PARALLEL}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}
MAMBA_SCHEDULER_STRATEGY=${MAMBA_SCHEDULER_STRATEGY}
ENABLE_SERVER_METRICS=${ENABLE_SERVER_METRICS}
DECODE_LOG_INTERVAL=${DECODE_LOG_INTERVAL}
METRICS_SCRAPE_INTERVAL=${METRICS_SCRAPE_INTERVAL}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
TEMPERATURE=${TEMPERATURE}
TOP_P=${TOP_P}
TOP_K=${TOP_K}
PRESENCE_PENALTY=${PRESENCE_PENALTY}
ENABLE_THINKING=${ENABLE_THINKING}
TREE=${TREE_LABEL}:${SPEC_NUM_STEPS}:${SPEC_TOPK}:${SPEC_DRAFT_TOKENS}
CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD}
CSD_HUGE_REBUILD_THRESHOLD=${CSD_HUGE_REBUILD_THRESHOLD}
PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH}
EOF

"${PYTHON}" - <<'PY' "${OOD_DATA_FILES}" | tee "${OUT_ROOT}/apps_taco_dataset_counts.txt" | tee -a "${OUT_ROOT}/apps_taco_config.txt"
import json
import sys
from pathlib import Path

print("DATASET_COUNTS:")
for path_str in sys.argv[1].split():
    path = Path(path_str)
    if path.suffix == ".jsonl":
        count = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
    else:
        count = len(json.load(path.open("r", encoding="utf-8")))
    print(f"{path.name}={count}")
PY

for data_file in ${OOD_DATA_FILES}; do
  if [[ ! -f "${data_file}" ]]; then
    echo "Data file not found: ${data_file}" >&2
    exit 1
  fi
  data_label=$(safe_name "$(basename "${data_file}" .json)")
  data_label=$(safe_name "${data_label%.jsonl}")
  dataset_name=$(safe_name "${data_label%%_*}")
  dataset_root="${OUT_ROOT}/${dataset_name}"
  answer_dir="${dataset_root}/answers"
  log_dir="${dataset_root}/logs"
  result_file="${dataset_root}/results.jsonl"
  mkdir -p "${answer_dir}" "${log_dir}"

  for method in ${METHOD_SET}; do
    run_name=$(safe_name "${method}_${data_label}_tree${TREE_LABEL}_${RUN_STAMP}")
    server_log="${log_dir}/${run_name}_server.log"
    bench_log="${log_dir}/${run_name}_bench.log"
    metrics_log="${log_dir}/${run_name}_decode_metrics.jsonl"
    answer_file="${answer_dir}/${run_name}_model_outputs.json"

    echo "------------------------------------------------------------"
    echo "Running method=${method}, data_file=${data_file}"
    echo "server_log=${server_log}"
    echo "bench_log=${bench_log}"
    echo "answer_file=${answer_file}"
    echo "------------------------------------------------------------"

    read -r -a server_args <<<"$(server_args_for_method "${method}")"
    "${PYTHON}" -m sglang.launch_server "${server_args[@]}" >"${server_log}" 2>&1 &
    server_pid=$!
    trap 'kill_server "${server_pid:-}"' EXIT
    wait_server

    metrics_pid=""
    if [[ "${ENABLE_SERVER_METRICS}" == "true" ]]; then
      "${PYTHON}" benchmark/csd/eval/scrape_sglang_decode_metrics.py \
        --host "${HOST}" \
        --port "${PORT}" \
        --interval "${METRICS_SCRAPE_INTERVAL}" \
        --output "${metrics_log}" \
        >"${metrics_log}.scraper.log" 2>&1 &
      metrics_pid=$!
    fi

    bench_args=(
      benchmark/csd/eval/bench_sglang_chat_generation.py
      --dataset alpaca_eval
      --task-name "${dataset_name}"
      --data-file "${data_file}"
      --host "${HOST}"
      --port "${PORT}"
      --backend srt
      --answer-file "${answer_file}"
      --model-id "${MODEL_ID}_${method}"
      --tokenizer-path "${TOKENIZER_PATH}"
      --parallel "${PARALLEL}"
      --result-file "${result_file}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --temperature "${TEMPERATURE}"
      --top-p "${TOP_P}"
      --top-k "${TOP_K}"
      --min-p "${MIN_P}"
      --presence-penalty "${PRESENCE_PENALTY}"
      --repetition-penalty "${REPETITION_PENALTY}"
      --system-prompt "${SYSTEM_PROMPT}"
      --enable-thinking "${ENABLE_THINKING}"
    )
    if [[ -n "${NUM_EXAMPLES}" ]]; then
      bench_args+=(--num-examples "${NUM_EXAMPLES}")
    fi

    set +e
    "${PYTHON}" "${bench_args[@]}" >"${bench_log}" 2>&1
    status=$?
    set -e
    if [[ -n "${metrics_pid}" ]]; then
      kill "${metrics_pid}" >/dev/null 2>&1 || true
      wait "${metrics_pid}" >/dev/null 2>&1 || true
    fi
    kill_server "${server_pid}"
    trap - EXIT

    if [[ "${status}" != "0" ]]; then
      echo "Domain OOD benchmark failed for method=${method}, data_file=${data_file}, status=${status}" >&2
      exit "${status}"
    fi
  done
  "${PYTHON}" benchmark/csd/eval/summarize_domain_ood_csd.py "${result_file}" \
    | tee "${dataset_root}/summary.md"
done

combined_result_file="${OUT_ROOT}/apps_taco_combined_results.jsonl"
: >"${combined_result_file}"
for data_file in ${OOD_DATA_FILES}; do
  data_label=$(safe_name "$(basename "${data_file}" .json)")
  data_label=$(safe_name "${data_label%.jsonl}")
  dataset_name=$(safe_name "${data_label%%_*}")
  cat "${OUT_ROOT}/${dataset_name}/results.jsonl" >>"${combined_result_file}"
done
"${PYTHON}" benchmark/csd/eval/summarize_domain_ood_csd.py "${combined_result_file}" \
  | tee "${OUT_ROOT}/apps_taco_combined_summary.md"

echo "Done."
echo "OUT_ROOT=${OUT_ROOT}"
