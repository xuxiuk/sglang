#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/sglang-dspark-csd}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/dspark_csd/v4_mtp314_calibration_dp4_20260802}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
TARGET_MODEL=${TARGET_MODEL:-/data/model/DeepSeek-V4-Flash}
MTP_DRAFT=${MTP_DRAFT:-/data/model/DeepSeek-V4-Flash-MTP-Draft}
PROMPTS=${PROMPTS:-/root/sglang-csd-archive-20260722/runs_legacy/redpajama/redpajama_csd_calibration_answer.jsonl}
GPU_SET=${GPU_SET:-4,5,6,7}
TP_SIZE=${TP_SIZE:-4}
DP_SIZE=${DP_SIZE:-4}
PARALLEL=${PARALLEL:-48}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-48}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
SWA_FULL_TOKENS_RATIO=${SWA_FULL_TOKENS_RATIO:-0.1}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-$((256 * DP_SIZE))}
PORT=${PORT:-32350}
DIST_INIT_ADDR=${DIST_INIT_ADDR:-127.0.0.1:32360}
NCCL_PORT=${NCCL_PORT:-32370}
SHARD_STEM=$RUN_ROOT/tables/v4_mtp314_redpajama_rank.json
MERGED_TABLE=$RUN_ROOT/tables/v4_mtp314_redpajama_merged.json
DELTA_CAPACITY=16777216

mkdir -p "$RUN_ROOT"/{logs,results,tables,metadata}
test -s "$TARGET_MODEL/config.json"
test -s "$MTP_DRAFT/config.json"
test -s "$PROMPTS"
test ! -e "$MERGED_TABLE" || {
  echo "Refusing to overwrite completed table: $MERGED_TABLE" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="$GPU_SET"
export NCCL_IB_DISABLE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset SGLANG_RAGGED_VERIFY_MODE
cd "$ROOT"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cat > "$RUN_ROOT/metadata/config.json" <<EOF
{
  "target_model": "$TARGET_MODEL",
  "draft_model": "$MTP_DRAFT",
  "algorithm": "EAGLE",
  "speculative_num_steps": 3,
  "speculative_eagle_topk": 1,
  "speculative_num_draft_tokens": 4,
  "tp_size": $TP_SIZE,
  "dp_size": $DP_SIZE,
  "gpu_set": "$GPU_SET",
  "dataset": "redpajama_6domains_n1000",
  "num_prompts": 6000,
  "max_new_tokens": 1024,
  "temperature": 1.0,
  "top_p": 1.0,
  "parallel": $PARALLEL,
  "csd_force_accept_disabled": true,
  "csd_dynamic_update_ignore_prob_ratio": true,
  "csd_freq_threshold": 6,
  "csd_delta_capacity": $DELTA_CAPACITY,
  "csd_rebuild_threshold": $DELTA_CAPACITY,
  "aggregation": "$DP_SIZE independent DP shards summed once after calibration"
}
EOF

echo STARTING_SERVER > "$RUN_ROOT/STATUS"
"$PYTHON" -m sglang.launch_server \
  --model-path "$TARGET_MODEL" \
  --speculative-draft-model-path "$MTP_DRAFT" \
  --tp "$TP_SIZE" --dp-size "$DP_SIZE" --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --cuda-graph-max-bs 64 --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --watchdog-timeout 7200 \
  --disable-radix-cache --trust-remote-code \
  --host 0.0.0.0 --port "$PORT" \
  --dist-init-addr "$DIST_INIT_ADDR" --nccl-port "$NCCL_PORT" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-csd \
  --speculative-csd-dynamic-update \
  --speculative-csd-dynamic-update-ignore-prob-ratio \
  --speculative-csd-force-accept-disabled \
  --speculative-csd-prob-ratio 0.01 \
  --speculative-csd-freq-threshold 6 \
  --speculative-csd-delta-capacity "$DELTA_CAPACITY" \
  --speculative-csd-rebuild-threshold "$DELTA_CAPACITY" \
  >"$RUN_ROOT/logs/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN_ROOT/server.pid"

ready=0
for _ in $(seq 1 240); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo SERVER_FAILED > "$RUN_ROOT/STATUS"
    wait "$SERVER_PID"
  fi
  sleep 5
done
test "$ready" -eq 1 || {
  echo SERVER_READY_TIMEOUT > "$RUN_ROOT/STATUS"
  exit 1
}

echo CALIBRATING > "$RUN_ROOT/STATUS"
"$PYTHON" scripts/calibrate_dspark_csd.py \
  --prompts "$PROMPTS" \
  --output "$RUN_ROOT/results/redpajama_calibration_answers.jsonl" \
  --summary "$RUN_ROOT/results/redpajama_calibration_summary.json" \
  --host 127.0.0.1 --port "$PORT" \
  --parallel "$PARALLEL" --max-new-tokens 1024 --temperature 1.0 --top-p 1.0 \
  2>&1 | tee "$RUN_ROOT/logs/calibration_client.log"

echo EXPORTING > "$RUN_ROOT/STATUS"
curl --noproxy '*' -fsS -X POST "http://127.0.0.1:$PORT/save_csd_table" \
  -H 'Content-Type: application/json' \
  -d "{\"path\":\"$SHARD_STEM\",\"metadata\":{\"dataset\":\"redpajama_6domains_n1000\",\"backend\":\"mtp_eagle\",\"target_model\":\"$TARGET_MODEL\",\"draft_model\":\"$MTP_DRAFT\",\"tree_shape\":\"3-1-4\",\"temperature\":1.0,\"top_p\":1.0}}" \
  | tee "$RUN_ROOT/logs/export_table.json"

mapfile -t shards < <(find "$RUN_ROOT/tables" -maxdepth 1 -name 'v4_mtp314_redpajama_rank.dp*.json' -type f | sort)
test "${#shards[@]}" -eq "$DP_SIZE" || {
  echo "Expected $DP_SIZE DP shards, got ${#shards[@]}" >&2
  exit 1
}

echo MERGING > "$RUN_ROOT/STATUS"
"$PYTHON" scripts/merge_csd_tables.py \
  --inputs "${shards[@]}" \
  --output "$MERGED_TABLE" \
  | tee "$RUN_ROOT/logs/merge_table.log"

"$PYTHON" - "$MERGED_TABLE" "${shards[@]}" <<'PY' \
  | tee "$RUN_ROOT/logs/validate_table.log"
import json
import sys
from pathlib import Path

merged_path = Path(sys.argv[1])
shards = [Path(p) for p in sys.argv[2:]]

def load(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

shard_payloads = [load(path) for path in shards]
merged = load(merged_path)
shard_freq_total = sum(
    int(entry.get("freq", 1))
    for payload in shard_payloads
    for entry in payload.get("entries", [])
)
merged_freq_total = sum(
    int(entry.get("freq", 1)) for entry in merged.get("entries", [])
)
if shard_freq_total != merged_freq_total:
    raise SystemExit(
        f"frequency mismatch: shards={shard_freq_total} merged={merged_freq_total}"
    )
summary = {
    "num_shards": len(shards),
    "shard_entries": [len(p.get("entries", [])) for p in shard_payloads],
    "shard_frequency_totals": [
        sum(int(e.get("freq", 1)) for e in p.get("entries", []))
        for p in shard_payloads
    ],
    "merged_unique_pairs": len(merged.get("entries", [])),
    "merged_frequency_total": merged_freq_total,
    "frequency_conservation": True,
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
with (merged_path.parent.parent / "results" / "table_validation.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
PY

curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/server_info" \
  > "$RUN_ROOT/results/server_info_final.json" || true
echo COMPLETE > "$RUN_ROOT/STATUS"
echo "Calibration complete: $MERGED_TABLE"
