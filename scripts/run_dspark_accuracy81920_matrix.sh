#!/usr/bin/env bash
# Run accuracy and throughput tasks for all DSpark+CSD modes. The avg4 task
# variants request four non-streaming completions per item and report both
# pass@1 averaged over four draws and pass@4. Sampling follows the
# DeepSeek-V4-Flash-DSpark recommendation. The 81920 output cap is an explicit
# experiment override. With DP=4 and client parallel=48, n=4 produces up to
# 192 live sequences, so the formal default reserves 48 slots per DP rank.

set -euo pipefail

ROOT=${ROOT:-/root/sglang-dspark-csd}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/dspark_csd/accuracy81920_matrix_$(date +%Y%m%d_%H%M%S)}
MODEL=${MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
CSD_TABLE=${CSD_TABLE:-$ROOT/runs/dspark_csd/formal_redpajama_20260724/tables/dspark_csd_merged.json}
ENTROPY_THRESHOLD=${ENTROPY_THRESHOLD:-1.5}
ENTROPY_MIN_THRESHOLD=${ENTROPY_MIN_THRESHOLD:--1}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-192}
EVAL_THREADS=${EVAL_THREADS:-48}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-86400}
MAX_RETRIES=${MAX_RETRIES:-0}
MAX_TOKENS=${MAX_TOKENS:-81920}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-96000}
MAX_SAMPLES=${MAX_SAMPLES:-}
GPU_SET=${GPU_SET:-4,5,6,7}
PORT=${PORT:-30000}
DIST_INIT_ADDR=${DIST_INIT_ADDR:-}
NCCL_PORT=${NCCL_PORT:-}
METHODS=${METHODS:-"bare plain dynamic entropy"}
TASKS=${TASKS:-"aime lcb"}
LCEVAL_PYTHON=${LCEVAL_PYTHON:-/root/miniconda3/envs/sglang/bin/python}
LIGHTEVAL_SRC=${LIGHTEVAL_SRC:-$ROOT/runs/mtp_csd/qwen35_accuracy_avg_314_c48_20260802/vendor/lighteval/src}
SGLANG_EVAL_SRC=${SGLANG_EVAL_SRC:-/root/sglang/python}
EVAL_PYTHONPATH=${EVAL_PYTHONPATH:-$LIGHTEVAL_SRC:$SGLANG_EVAL_SRC}
TACO_DATA=${TACO_DATA:-}
APPS_DATA=${APPS_DATA:-}
CODE_OOD_MAX_EXAMPLES=${CODE_OOD_MAX_EXAMPLES:-1000}
CODE_OOD_NUM_SAMPLES=${CODE_OOD_NUM_SAMPLES:-4}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-dspark-csd-cu128
cd "$ROOT"

# Local evaluation traffic must never be sent through the machine's outbound
# HTTP proxy.  Both spellings are needed because clients differ in which one
# they honor.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

test -s "$CSD_TABLE"
test -x "$LCEVAL_PYTHON"
test -d "$LIGHTEVAL_SRC/lighteval"
mkdir -p "$RUN_ROOT"/{driver_logs,logs,results,tables}
test ! -e "$RUN_ROOT/config.txt" || {
  echo "Refusing to reuse run directory: $RUN_ROOT" >&2
  exit 1
}

cat >"$RUN_ROOT/config.txt" <<EOF
started_at=$(date -Is)
git_head=$(git rev-parse HEAD)
model=$MODEL
methods=$METHODS
tasks=$TASKS
temperature=1.0
top_p=1.0
aime_thinking=true
lcb_reasoning_mode=thinking_high
chat_encoding=sglang.encoding_dsv4
thinking=true
reasoning_effort=high
max_tokens=$MAX_TOKENS
max_model_length=$MAX_MODEL_LENGTH
max_samples=${MAX_SAMPLES:-all}
gpu_set=$GPU_SET
eval_threads=$EVAL_THREADS
parallel_requests=$EVAL_THREADS
max_running_requests=$MAX_RUNNING_REQUESTS
request_timeout_seconds=$REQUEST_TIMEOUT_SECONDS
max_retries_after_initial_attempt=$MAX_RETRIES
litellm_num_retries=0
server_lifecycle=restart_per_task_and_method
http_port=$PORT
dist_init_addr=${DIST_INIT_ADDR:-auto}
nccl_port=${NCCL_PORT:-auto}
csd_table=$CSD_TABLE
csd_table_sha256=$(sha256sum "$CSD_TABLE" | awk '{print $1}')
csd_prob_ratio=$CSD_PROB_RATIO
entropy_threshold=$ENTROPY_THRESHOLD
entropy_min_threshold=$ENTROPY_MIN_THRESHOLD
code_ood_max_examples=$CODE_OOD_MAX_EXAMPLES
code_ood_num_samples=$CODE_OOD_NUM_SAMPLES
EOF

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap cleanup_server EXIT INT TERM

wait_ready() {
  for _ in $(seq 1 3600); do
    if curl --noproxy '*' -fsS --max-time 5 \
      "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1; then
      return 0
    fi
    kill -0 "$server_pid" 2>/dev/null || {
      echo "Server exited before readiness" >&2
      return 1
    }
    sleep 1
  done
  echo "Timed out waiting for server" >&2
  return 1
}

snapshot_server_info() {
  local output=$1
  curl --noproxy '*' -fsS --max-time 60 \
    "http://127.0.0.1:$PORT/server_info" >"$output"
  "$LCEVAL_PYTHON" -m json.tool "$output" >/dev/null
}

run_scored_task() {
  local task=$1
  local pair_root=$2
  local method=$3
  local eval_task output_name

  case "$task" in
    aime25_avg4) eval_task=aime25_avg4; output_name=aime25_avg4 ;;
    math500_avg4) eval_task=math500_avg4; output_name=math500_avg4 ;;
    lcb_avg4) eval_task=lcb; output_name=lcb_codegen_v6_avg4 ;;
    gsm8k_avg4) eval_task=gsm8k_avg4; output_name=gsm8k_avg4 ;;
    *) return 2 ;;
  esac

  local sample_args=()
  if [[ -n "$MAX_SAMPLES" ]]; then
    sample_args+=(--max-samples "$MAX_SAMPLES")
  fi

  env PYTHONPATH="${EVAL_PYTHONPATH:-${PYTHONPATH:-}}" \
    "$LCEVAL_PYTHON" scripts/eval_dspark_long.py \
    --task "$eval_task" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL" \
    --output-dir "$pair_root/eval" \
    --parallel "$EVAL_THREADS" \
    --max-new-tokens "$MAX_TOKENS" \
    --max-model-length "$MAX_MODEL_LENGTH" \
    "${sample_args[@]}" \
    --temperature 1.0 --top-p 1.0 \
    --thinking --reasoning-effort high \
    --timeout "$REQUEST_TIMEOUT_SECONDS" \
    --max-retries "$MAX_RETRIES" \
    --request-timing-file "$pair_root/eval/request_timing.jsonl" \
    2>&1 | tee "$RUN_ROOT/driver_logs/eval_${method}_${output_name}.log"
}

run_code_ood_task() {
  local task=$1
  local pair_root=$2
  local method=$3
  local data_file
  case "$task" in
    taco) data_file=$TACO_DATA ;;
    apps) data_file=$APPS_DATA ;;
    *) return 2 ;;
  esac
  test -s "$data_file"
  python scripts/eval_dspark_code_ood.py \
    --task "$task" \
    --data-file "$data_file" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --output-dir "$pair_root/eval" \
    --max-examples "$CODE_OOD_MAX_EXAMPLES" \
    --num-samples "$CODE_OOD_NUM_SAMPLES" \
    --parallel "$EVAL_THREADS" \
    --max-new-tokens "$MAX_TOKENS" \
    --temperature 1.0 --top-p 1.0 \
    2>&1 | tee "$RUN_ROOT/driver_logs/eval_${method}_${task}.log"
}

for task in $TASKS; do
for method in $METHODS; do
  case "$method" in
    bare) server_mode=bare-server ;;
    plain) server_mode=plain-server ;;
    dynamic) server_mode=dynamic-server ;;
    entropy) server_mode=entropy-server ;;
    entropy_min) server_mode=entropy-min-server ;;
    *) echo "Unknown method: $method" >&2; exit 2 ;;
  esac

  pair_root="$RUN_ROOT/$method/$task"
  server_root="$pair_root/server"
  mkdir -p "$server_root/logs" "$pair_root/metrics"
  echo "[$(date -Is)] start task=$task method=$method" | tee -a "$RUN_ROOT/driver_logs/supervisor.log"

  setsid env \
    ROOT="$ROOT" RUN_ROOT="$server_root" MODEL="$MODEL" PORT="$PORT" \
    GPU_SET="$GPU_SET" NCCL_NET="${NCCL_NET:-Socket}" \
    DIST_INIT_ADDR="$DIST_INIT_ADDR" NCCL_PORT="$NCCL_PORT" \
    MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS" CSD_TABLE="$CSD_TABLE" \
    CSD_PROB_RATIO="$CSD_PROB_RATIO" ENTROPY_THRESHOLD="$ENTROPY_THRESHOLD" \
    ENTROPY_MIN_THRESHOLD="$ENTROPY_MIN_THRESHOLD" \
    SGLANG_DEFAULT_THINKING=true SGLANG_DSV4_REASONING_EFFORT=high \
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
    bash scripts/run_dspark_csd.sh "$server_mode" \
    >"$RUN_ROOT/driver_logs/server_${method}_${task}.log" 2>&1 &
  server_pid=$!
  wait_ready
  snapshot_server_info "$pair_root/metrics/server_info_baseline.json"

  eval_status=0
  if [[ "$task" == "aime" ]]; then
    sgl-eval run aime25 \
      --base-url "http://127.0.0.1:$PORT/v1" \
      --model "$MODEL" \
      --n-repeats 1 \
      --temperature 1.0 --top-p 1.0 --thinking \
      --max-tokens "$MAX_TOKENS" --num-threads "$EVAL_THREADS" \
      --out-dir "$pair_root/eval" \
      2>&1 | tee "$RUN_ROOT/driver_logs/eval_${method}_aime25.log"
  elif [[ "$task" == "lcb" ]]; then
    env PYTHONPATH="${EVAL_PYTHONPATH:-${PYTHONPATH:-}}" \
      "$LCEVAL_PYTHON" scripts/eval_dspark_long.py \
      --task lcb \
      --base-url "http://127.0.0.1:$PORT/v1" \
      --model "$MODEL" \
      --output-dir "$pair_root/eval" \
      --parallel "$EVAL_THREADS" \
      --max-new-tokens "$MAX_TOKENS" \
      --max-model-length "$MAX_MODEL_LENGTH" \
      --temperature 1.0 --top-p 1.0 \
      --thinking --reasoning-effort high \
      --timeout "$REQUEST_TIMEOUT_SECONDS" --max-retries "$MAX_RETRIES" \
      --request-timing-file "$pair_root/eval/request_timing.jsonl" \
      2>&1 | tee "$RUN_ROOT/driver_logs/eval_${method}_lcb.log"
  elif [[ "$task" == "taco" || "$task" == "apps" ]]; then
    run_code_ood_task "$task" "$pair_root" "$method" || eval_status=$?
  else
    run_scored_task "$task" "$pair_root" "$method" || eval_status=$?
  fi

  snapshot_server_info "$pair_root/metrics/server_info_final.json"
  "$LCEVAL_PYTHON" scripts/summarize_dspark_task_metrics.py \
    --baseline "$pair_root/metrics/server_info_baseline.json" \
    --final "$pair_root/metrics/server_info_final.json" \
    --output "$pair_root/metrics/task_metrics.json"
  echo "[$(date -Is)] complete task=$task method=$method status=$eval_status" \
    | tee -a "$RUN_ROOT/driver_logs/supervisor.log"
  cleanup_server
  if [[ "$eval_status" -ne 0 ]]; then
    echo "Evaluation failed without retry: task=$task method=$method status=$eval_status" >&2
    exit "$eval_status"
  fi
done
done

echo "completed_at=$(date -Is)" | tee -a "$RUN_ROOT/config.txt"
echo "RUN_ROOT=$RUN_ROOT"
