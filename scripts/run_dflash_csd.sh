#!/usr/bin/env bash
set -euo pipefail

# DFlash + unified CSD server/calibration entry point.  Outputs are isolated
# from the archived v0.5.11 DFlash repository.
ROOT=${ROOT:-/root/sglang-dspark-csd}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/dflash_csd}
MODEL=${MODEL:-/data/model/Qwen3.5-35B-A3B}
DRAFT_MODEL=${DRAFT_MODEL:-/data/model/Qwen3.5-35B-A3B-DFlash}
REDPAJAMA_PROMPTS=${REDPAJAMA_PROMPTS:-/root/sglang-csd-archive-20260722/runs_legacy/redpajama/redpajama_csd_calibration_answer.jsonl}
CSD_TABLE=${CSD_TABLE:-$RUN_ROOT/tables/dflash_block16.json}
GPU_SET=${GPU_SET:-0,1,2,3}
PORT=${PORT:-31200}
NCCL_PORT=${NCCL_PORT:-}
BLOCK_SIZE=${BLOCK_SIZE:-16}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
CSD_DELTA_CAPACITY=${CSD_DELTA_CAPACITY:-16777216}
CSD_PROB_RATIO=${CSD_PROB_RATIO:-0.3}
CSD_REBUILD_THRESHOLD=${CSD_REBUILD_THRESHOLD:-4096}
ENTROPY_THRESHOLD=${ENTROPY_THRESHOLD:--1}
ENTROPY_MIN_THRESHOLD=${ENTROPY_MIN_THRESHOLD:--1}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-dspark-csd-cu128
cd "$ROOT"
mkdir -p "$RUN_ROOT"/{logs,results,tables}

export CUDA_VISIBLE_DEVICES="$GPU_SET"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export PATH="$CONDA_PREFIX/bin:$PATH"

common_server_args=(
  --model-path "$MODEL"
  --tokenizer-path "$MODEL"
  --tp-size 4 --trust-remote-code
  --mem-fraction-static 0.75
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --host 0.0.0.0 --port "$PORT"
  --disable-radix-cache
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "$DRAFT_MODEL"
  --speculative-dflash-block-size "$BLOCK_SIZE"
)
if [[ -n "$NCCL_PORT" ]]; then
  common_server_args+=(--nccl-port "$NCCL_PORT")
fi

serve() {
  local method=$1
  shift
  test -s "$MODEL/config.json"
  test -s "$DRAFT_MODEL/config.json"
  python -m sglang.launch_server "${common_server_args[@]}" "$@" \
    2>&1 | tee "$RUN_ROOT/logs/server_${method}.log"
}

case "${1:-}" in
  bare-server)
    serve bare
    ;;
  calibration-server)
    serve calibration \
      --speculative-csd \
      --speculative-csd-dynamic-update \
      --speculative-csd-dynamic-update-ignore-prob-ratio \
      --speculative-csd-force-accept-disabled \
      --speculative-csd-prob-ratio 0.01 \
      --speculative-csd-freq-threshold 6 \
      --speculative-csd-delta-capacity "$CSD_DELTA_CAPACITY" \
      --speculative-csd-rebuild-threshold "$CSD_REBUILD_THRESHOLD"
    ;;
  calibration-run)
    test -s "$REDPAJAMA_PROMPTS"
    python scripts/calibrate_dspark_csd.py \
      --prompts "$REDPAJAMA_PROMPTS" \
      --output "$RUN_ROOT/results/redpajama_calibration_answers.jsonl" \
      --summary "$RUN_ROOT/results/redpajama_calibration_summary.json" \
      --host 127.0.0.1 --port "$PORT" \
      --parallel 48 --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 \
      2>&1 | tee "$RUN_ROOT/logs/redpajama_calibration_client.log"
    ;;
  export-table)
    test ! -e "$CSD_TABLE" || {
      echo "Refusing to overwrite $CSD_TABLE" >&2
      exit 1
    }
    curl -fsS -X POST "http://127.0.0.1:$PORT/save_csd_table" \
      -H 'Content-Type: application/json' \
      -d "{\"path\":\"$CSD_TABLE\",\"metadata\":{\"dataset\":\"redpajama_6domains_n1000\",\"temperature\":1.0,\"backend\":\"dflash\",\"block_size\":$BLOCK_SIZE}}" \
      | tee "$RUN_ROOT/logs/export_table.json"
    ;;
  plain-server)
    test -s "$CSD_TABLE"
    serve plain \
      --speculative-csd \
      --speculative-csd-table-path "$CSD_TABLE" \
      --speculative-csd-freq-threshold 6 \
      --speculative-csd-prob-ratio "$CSD_PROB_RATIO"
    ;;
  dynamic-server)
    test -s "$CSD_TABLE"
    serve dynamic \
      --speculative-csd \
      --speculative-csd-table-path "$CSD_TABLE" \
      --speculative-csd-freq-threshold 6 \
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
      --speculative-csd \
      --speculative-csd-table-path "$CSD_TABLE" \
      --speculative-csd-freq-threshold 6 \
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
      --speculative-csd \
      --speculative-csd-table-path "$CSD_TABLE" \
      --speculative-csd-freq-threshold 6 \
      --speculative-csd-prob-ratio "$CSD_PROB_RATIO" \
      --speculative-csd-dynamic-update \
      --speculative-csd-dynamic-update-ignore-prob-ratio \
      --speculative-csd-delta-capacity "$CSD_DELTA_CAPACITY" \
      --speculative-csd-rebuild-threshold "$CSD_REBUILD_THRESHOLD" \
      --speculative-csd-force-accept-entropy-min-threshold "$ENTROPY_MIN_THRESHOLD"
    ;;
  metrics)
    curl -fsS "http://127.0.0.1:$PORT/server_info" \
      | python -m json.tool \
      | tee "$RUN_ROOT/logs/csd_metrics_$(date +%Y%m%d_%H%M%S).json"
    ;;
  *)
    echo "Usage: $0 {bare-server|calibration-server|calibration-run|export-table|plain-server|dynamic-server|entropy-server|entropy-min-server|metrics}" >&2
    exit 2
    ;;
esac
