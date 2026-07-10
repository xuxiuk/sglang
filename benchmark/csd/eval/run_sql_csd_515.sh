#!/usr/bin/env bash
# Run text-to-SQL generation throughput/spec metrics with the 515 CSD setup.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_sql_csd_515}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/benchmark/csd/runs/sql_csd_515/${RUN_STAMP}"}
RESULT_FILE=${RESULT_FILE:-"${OUT_ROOT}/results/sql_csd_515.jsonl"}
ANSWER_DIR=${ANSWER_DIR:-"${OUT_ROOT}/answers"}
LOG_DIR=${LOG_DIR:-"${OUT_ROOT}/logs"}
mkdir -p "${OUT_ROOT}" "${ANSWER_DIR}" "${LOG_DIR}" "$(dirname "${RESULT_FILE}")"

PYTHON=${PYTHON:-/root/miniconda3/envs/sglang/bin/python}
CONDA_BIN=$(dirname "${PYTHON}")
MODEL_PATH=${MODEL_PATH:-/root/model/Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
MODEL_ID=${MODEL_ID:-Qwen3.5-35B-A3B}
CUDA_DEVICES=${CUDA_DEVICES:-4,5,6,7}
TP_SIZE=${TP_SIZE:-4}
PORT=${PORT:-30064}
HOST=${HOST:-127.0.0.1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.72}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-24}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-7200}
PARALLEL=${PARALLEL:-24}

DATASET=${DATASET:-sql_create_context}
NUM_EXAMPLES=${NUM_EXAMPLES:-500}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
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
ENTROPY_P20_THRESHOLD=${ENTROPY_P20_THRESHOLD:-1.5638477802276611}
PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH:-"${REPO_ROOT}/benchmark/csd/runs/redpajama/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_temp1.0_ratio0.01_maxnew1024.json"}

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
    --trust-remote-code
    --disable-radix-cache
    --disable-overlap-schedule
    --log-level warning
  )

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
    dynamic_ignore_ratio|dynamic_entropy_p20_ignore_ratio)
      args+=(
        --speculative-csd
        --speculative-csd-table-path "${PLAIN_CSD_TABLE_PATH}"
        --speculative-csd-freq-threshold "${CSD_FREQ_THRESHOLD}"
        --speculative-csd-key-selection-strategy "${CSD_KEY_SELECTION_STRATEGY}"
        --speculative-csd-score-threshold "${CSD_SCORE_THRESHOLD}"
        --speculative-csd-prob-ratio "${CSD_PROB_RATIO}"
        --speculative-csd-dynamic-update
        --speculative-csd-rebuild-threshold "${CSD_REBUILD_THRESHOLD}"
        --speculative-csd-dynamic-update-ignore-prob-ratio
      )
      if [[ "${method}" == "dynamic_entropy_p20_ignore_ratio" ]]; then
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

cat <<EOF | tee "${OUT_ROOT}/run_config.txt"
OUT_ROOT=${OUT_ROOT}
RESULT_FILE=${RESULT_FILE}
DATASET=${DATASET}
NUM_EXAMPLES=${NUM_EXAMPLES}
MODEL_PATH=${MODEL_PATH}
CUDA_DEVICES=${CUDA_DEVICES}
TP_SIZE=${TP_SIZE}
PORT=${PORT}
METHOD_SET=${METHOD_SET}
PARALLEL=${PARALLEL}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
TEMPERATURE=${TEMPERATURE}
TOP_P=${TOP_P}
TOP_K=${TOP_K}
PRESENCE_PENALTY=${PRESENCE_PENALTY}
ENABLE_THINKING=${ENABLE_THINKING}
TREE=${TREE_LABEL}:${SPEC_NUM_STEPS}:${SPEC_TOPK}:${SPEC_DRAFT_TOKENS}
PLAIN_CSD_TABLE_PATH=${PLAIN_CSD_TABLE_PATH}
EOF

for method in ${METHOD_SET}; do
  run_name=$(safe_name "${method}_${DATASET}_tree${TREE_LABEL}_${RUN_STAMP}")
  server_log="${LOG_DIR}/${run_name}_server.log"
  bench_log="${LOG_DIR}/${run_name}_bench.log"
  answer_file="${ANSWER_DIR}/${run_name}_answers.jsonl"

  echo "------------------------------------------------------------"
  echo "Running method=${method}, dataset=${DATASET}"
  echo "server_log=${server_log}"
  echo "bench_log=${bench_log}"
  echo "answer_file=${answer_file}"
  echo "------------------------------------------------------------"

  read -r -a server_args <<<"$(server_args_for_method "${method}")"
  "${PYTHON}" -m sglang.launch_server "${server_args[@]}" >"${server_log}" 2>&1 &
  server_pid=$!
  trap 'kill_server "${server_pid:-}"' EXIT
  wait_server

  set +e
  "${PYTHON}" benchmark/csd/eval/bench_sglang_sql_generation.py \
    --dataset "${DATASET}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --backend srt \
    --answer-file "${answer_file}" \
    --result-file "${RESULT_FILE}" \
    --model-id "${MODEL_ID}_${method}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --parallel "${PARALLEL}" \
    --num-examples "${NUM_EXAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --min-p "${MIN_P}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    --repetition-penalty "${REPETITION_PENALTY}" \
    --system-prompt "${SYSTEM_PROMPT}" \
    --enable-thinking "${ENABLE_THINKING}" \
    >"${bench_log}" 2>&1
  status=$?
  set -e
  kill_server "${server_pid}"
  trap - EXIT

  if [[ "${status}" != "0" ]]; then
    echo "SQL benchmark failed for method=${method}, status=${status}" >&2
    exit "${status}"
  fi
done

echo "Done."
echo "OUT_ROOT=${OUT_ROOT}"
echo "RESULT_FILE=${RESULT_FILE}"
