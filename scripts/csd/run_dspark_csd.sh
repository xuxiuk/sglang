#!/usr/bin/env bash
set -euo pipefail

# DSpark + CSD calibration and evaluation. Every output is isolated under
# runs/dspark_csd; existing DSpark blog reproduction results are never touched.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/root/sglang-dspark-csd}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/dspark_csd}
MODEL=${MODEL:-/data/model/DeepSeek-V4-Flash-DSpark}
SPS_TABLE=${SPS_TABLE:-$ROOT/runs/dspark_blog_repro/profile/dspark_sps_additive_h20_dp4_bs64.json}
PROMPT=${PROMPT:-$ROOT/runs/dspark_blog_repro/prompts/frontier_prompt.txt}
REDPAJAMA_PROMPTS=${REDPAJAMA_PROMPTS:-/root/sglang-csd-archive-20260722/runs_legacy/redpajama/redpajama_csd_calibration_answer.jsonl}
CSD_TABLE=${CSD_TABLE:-$RUN_ROOT/tables/dspark_csd_merged.json}
GPU_SET=${GPU_SET:-0,1,2,3,4,5,6,7}
TP_SIZE=${TP_SIZE:-8}
DP_SIZE=${DP_SIZE:-8}
EP_SIZE=${EP_SIZE:-1}
ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-1}
ENABLE_DP_LM_HEAD=${ENABLE_DP_LM_HEAD:-$ENABLE_DP_ATTENTION}
MOE_A2A_BACKEND=${MOE_A2A_BACKEND:-none}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}
DEEPEP_MODE=${DEEPEP_MODE:-auto}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-$((256 * DP_SIZE))}
PORT=${PORT:-30000}
DIST_INIT_ADDR=${DIST_INIT_ADDR:-}
NCCL_PORT=${NCCL_PORT:-}
ENTROPY_THRESHOLD=${ENTROPY_THRESHOLD:--1}
ENTROPY_MIN_THRESHOLD=${ENTROPY_MIN_THRESHOLD:--1}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
SWA_FULL_TOKENS_RATIO=${SWA_FULL_TOKENS_RATIO:-0.2}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
SWA_EVICTION_INTERVAL=${SWA_EVICTION_INTERVAL:-128}
CSD_DELTA_CAPACITY=${CSD_DELTA_CAPACITY:-16777216}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD:-4096}
RAGGED_VERIFY_MODE=${RAGGED_VERIFY_MODE:-compact}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-dspark-csd-cu128
cd "$ROOT"
mkdir -p "$RUN_ROOT"/{logs,results,tables}

export CUDA_VISIBLE_DEVICES="$GPU_SET"
export NCCL_IB_DISABLE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export SGLANG_RAGGED_VERIFY_MODE="$RAGGED_VERIFY_MODE"
export SGLANG_SWA_EVICTION_INTERVAL="$SWA_EVICTION_INTERVAL"

common_server_args=(
  --model-path "$MODEL"
  --tp "$TP_SIZE" --dp-size "$DP_SIZE" --ep-size "$EP_SIZE"
  --moe-a2a-backend "$MOE_A2A_BACKEND" --moe-runner-backend "$MOE_RUNNER_BACKEND"
  --disable-flashinfer-autotune --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC"
  --cuda-graph-max-bs 64 --max-running-requests "$MAX_RUNNING_REQUESTS"
  --disable-radix-cache --trust-remote-code
  --host 0.0.0.0 --port "$PORT"
)

if [[ "$ENABLE_DP_ATTENTION" == 1 ]]; then
  common_server_args+=(--enable-dp-attention)
fi
if [[ "$ENABLE_DP_LM_HEAD" == 1 ]]; then
  common_server_args+=(--enable-dp-lm-head)
fi
if [[ "$MOE_A2A_BACKEND" == deepep ]]; then
  common_server_args+=(--deepep-mode "$DEEPEP_MODE")
fi

if [[ "$ENABLE_DP_ATTENTION" == 1 && "$MOE_A2A_BACKEND" != none ]]; then
  echo "DSpark DP attention requires MOE_A2A_BACKEND=none in this commit" >&2
  exit 2
fi
if [[ "$ENABLE_DP_ATTENTION" == 1 && "$DP_SIZE" -gt "$TP_SIZE" ]]; then
  echo "DP_SIZE cannot exceed TP_SIZE when DP attention is enabled" >&2
  exit 2
fi

# Multiple four-GPU servers can coexist on one host only when their internal
# distributed-control ports are disjoint.  Leave these unset for the normal
# single-server case; matrix runs may assign them explicitly.
if [[ -n "$DIST_INIT_ADDR" ]]; then
  common_server_args+=(--dist-init-addr "$DIST_INIT_ADDR")
fi
if [[ -n "$NCCL_PORT" ]]; then
  common_server_args+=(--nccl-port "$NCCL_PORT")
fi

serve() {
  local method=$1
  shift
  test -s "$SPS_TABLE"
  python -m sglang.launch_server "${common_server_args[@]}" \
    --speculative-algorithm DSPARK \
    --speculative-dspark-sps-table-path "$SPS_TABLE" "$@" \
    2>&1 | tee "$RUN_ROOT/logs/server_${method}.log"
}

serve_auto() {
  python -m sglang.launch_server "${common_server_args[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/server_auto.log"
}

benchmark() {
  local method=$1
  local batch_sizes=$2
  local output_len=$3
  local output="$RUN_ROOT/results/${method}.jsonl"
  test ! -e "$output" || { echo "Refusing to overwrite $output" >&2; exit 1; }
  # shellcheck disable=SC2086
  python -m sglang.benchmark.one_batch_server \
    --model None --base-url "http://127.0.0.1:$PORT" \
    --batch-size $batch_sizes --output-len "$output_len" --temperature 0.7 \
    --fixed-prompt-file "$PROMPT" --fixed-prompt-apply-chat-template \
    --skip-warmup --show-report --result-filename "$output" \
    2>&1 | tee "$RUN_ROOT/logs/bench_${method}.log"
}

case "${1:-}" in
  auto-server)
    serve_auto
    ;;
  calibration-server)
    serve calibration \
      --speculative-csd \
      --speculative-csd-dynamic-update \
      --speculative-csd-dynamic-update-ignore-prob-ratio \
      --speculative-csd-force-accept-disabled \
      --speculative-csd-freq-threshold 6 \
      --speculative-csd-delta-capacity "$CSD_DELTA_CAPACITY" \
      --speculative-csd-rebuild-threshold "$CSD_REBUILD_THRESHOLD"
    ;;
  formal-calibration-server)
    test -s "$REDPAJAMA_PROMPTS"
    serve formal_calibration \
      --speculative-csd \
      --speculative-csd-dynamic-update \
      --speculative-csd-dynamic-update-ignore-prob-ratio \
      --speculative-csd-force-accept-disabled \
      --speculative-csd-prob-ratio 0.01 \
      --speculative-csd-freq-threshold 6 \
      --speculative-csd-delta-capacity "$CSD_DELTA_CAPACITY" \
      --speculative-csd-rebuild-threshold "$CSD_REBUILD_THRESHOLD"
    ;;
  formal-calibration-run)
    test -s "$REDPAJAMA_PROMPTS"
    python "$SCRIPT_DIR/calibrate_dspark_csd.py" \
      --prompts "$REDPAJAMA_PROMPTS" \
      --output "$RUN_ROOT/results/redpajama_calibration_answers.jsonl" \
      --summary "$RUN_ROOT/results/redpajama_calibration_summary.json" \
      --host 127.0.0.1 --port "$PORT" \
      --parallel 8 --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 \
      2>&1 | tee "$RUN_ROOT/logs/redpajama_calibration_client.log"
    ;;
  calibration-smoke)
    benchmark calibration_smoke "8" 128
    ;;
  calibration-full)
    benchmark calibration_full "8 32 64 128 192 256" 1024
    ;;
  export-table)
    if compgen -G "$RUN_ROOT/tables/dspark_csd_rank.dp*.json" >/dev/null; then
      echo "Refusing to overwrite existing rank table shards" >&2
      exit 1
    fi
    curl -fsS -X POST "http://127.0.0.1:$PORT/save_csd_table" \
      -H 'Content-Type: application/json' \
      -d "{\"path\":\"$RUN_ROOT/tables/dspark_csd_rank.json\",\"metadata\":{\"dataset\":\"${CALIBRATION_DATASET:-redpajama_6domains_n1000}\",\"temperature\":${CALIBRATION_TEMPERATURE:-1.0}}}" \
      | tee "$RUN_ROOT/logs/export_table.json"
    mapfile -t shards < <(find "$RUN_ROOT/tables" -name 'dspark_csd_rank.dp*.json' -type f | sort)
    test "${#shards[@]}" -eq "$DP_SIZE" || {
      echo "Expected $DP_SIZE DP table shards, found ${#shards[@]}" >&2
      exit 1
    }
    python "$SCRIPT_DIR/merge_csd_tables.py" --inputs "${shards[@]}" --output "$CSD_TABLE"
    ;;
  metrics)
    curl -fsS "http://127.0.0.1:$PORT/server_info" \
      | python -m json.tool | tee "$RUN_ROOT/logs/csd_metrics_$(date +%Y%m%d_%H%M%S).json"
    ;;
  bare-server)
    serve bare
    ;;
  plain-server)
    test -s "$CSD_TABLE"
    serve plain \
      --speculative-csd --speculative-csd-table-path "$CSD_TABLE" \
      --speculative-csd-prob-ratio "$CSD_PROB_RATIO"
    ;;
  dynamic-server)
    test -s "$CSD_TABLE"
    serve dynamic \
      --speculative-csd --speculative-csd-table-path "$CSD_TABLE" \
      --speculative-csd-prob-ratio "$CSD_PROB_RATIO" \
      --speculative-csd-dynamic-update \
      --speculative-csd-dynamic-update-ignore-prob-ratio \
      --speculative-csd-delta-capacity "$CSD_DELTA_CAPACITY" \
      --speculative-csd-rebuild-threshold "$CSD_REBUILD_THRESHOLD"
    ;;
  entropy-server)
    test -s "$CSD_TABLE"
    test "$ENTROPY_THRESHOLD" != -1 || {
      echo "Set ENTROPY_THRESHOLD to a calibrated value" >&2
      exit 1
    }
    serve dynamic_entropy \
      --speculative-csd --speculative-csd-table-path "$CSD_TABLE" \
      --speculative-csd-prob-ratio "$CSD_PROB_RATIO" \
      --speculative-csd-dynamic-update \
      --speculative-csd-dynamic-update-ignore-prob-ratio \
      --speculative-csd-delta-capacity "$CSD_DELTA_CAPACITY" \
      --speculative-csd-rebuild-threshold "$CSD_REBUILD_THRESHOLD" \
      --speculative-csd-force-accept-entropy-threshold "$ENTROPY_THRESHOLD"
    ;;
  entropy-min-server)
    test -s "$CSD_TABLE"
    test "$ENTROPY_MIN_THRESHOLD" != -1 || {
      echo "Set ENTROPY_MIN_THRESHOLD to a calibrated value" >&2
      exit 1
    }
    serve dynamic_entropy_min \
      --speculative-csd --speculative-csd-table-path "$CSD_TABLE" \
      --speculative-csd-prob-ratio "$CSD_PROB_RATIO" \
      --speculative-csd-dynamic-update \
      --speculative-csd-dynamic-update-ignore-prob-ratio \
      --speculative-csd-delta-capacity "$CSD_DELTA_CAPACITY" \
      --speculative-csd-rebuild-threshold "$CSD_REBUILD_THRESHOLD" \
      --speculative-csd-force-accept-entropy-min-threshold "$ENTROPY_MIN_THRESHOLD"
    ;;
  smoke-benchmark)
    benchmark "${METHOD:-smoke}" "8" 128
    ;;
  curve-benchmark)
    benchmark "${METHOD:?Set METHOD}" "1 8 16 32 64 96 128 192 256" 1024
    ;;
  *)
    echo "Usage: $0 {calibration-server|formal-calibration-server|formal-calibration-run|calibration-smoke|calibration-full|export-table|metrics|bare-server|plain-server|dynamic-server|entropy-server|entropy-min-server|smoke-benchmark|curve-benchmark}" >&2
    exit 2
    ;;
esac
