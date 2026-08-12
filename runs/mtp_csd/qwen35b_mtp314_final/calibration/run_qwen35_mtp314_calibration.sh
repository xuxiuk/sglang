#!/usr/bin/env bash
# Rebuild the Qwen3.5 MTP 3-1-4 RedPajama CSD table used by this run.
# Historical CLI metadata records num_draft_tokens=3; with the verifier root
# token included, this is the effective 4-token width denoted by 3-1-4 here.
set -euo pipefail

ROOT=${ROOT:-/root/sglang-dspark-csd}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}
MODEL=${MODEL:-/data/model/Qwen3.5-35B-A3B}
PROMPTS=${PROMPTS:-/root/sglang-csd-archive-20260722/runs_legacy/redpajama/redpajama_csd_calibration_answer.jsonl}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-$SCRIPT_DIR/reproduced_mtp314_$RUN_STAMP}
TABLE=${TABLE:-$OUT_ROOT/tables/qwen35_mtp314_redpajama.json}
GPU_SET=${GPU_SET:-0,1,2,3}
PORT=${PORT:-32400}
NCCL_PORT=${NCCL_PORT:-32401}

mkdir -p "$OUT_ROOT"/{logs,results,tables,metadata}
test -s "$MODEL/config.json"
test -s "$PROMPTS"
test ! -e "$TABLE" || {
  echo "Refusing to overwrite existing table: $TABLE" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="$GPU_SET"
export NCCL_IB_DISABLE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
cd "$ROOT"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 4 --trust-remote-code \
  --mem-fraction-static 0.75 \
  --mamba-scheduler-strategy extra_buffer \
  --max-running-requests 48 \
  --watchdog-timeout 7200 \
  --host 0.0.0.0 --port "$PORT" --nccl-port "$NCCL_PORT" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --speculative-csd \
  --speculative-csd-dynamic-update \
  --speculative-csd-dynamic-update-ignore-prob-ratio \
  --speculative-csd-force-accept-disabled \
  --speculative-csd-prob-ratio 0.01 \
  --speculative-csd-freq-threshold 3 \
  --speculative-csd-delta-capacity 16777216 \
  --speculative-csd-rebuild-threshold 16777216 \
  >"$OUT_ROOT/logs/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT_ROOT/server.pid"

ready=0
for _ in $(seq 1 1440); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Calibration server exited; see $OUT_ROOT/logs/server.log" >&2
    wait "$SERVER_PID"
  fi
  sleep 5
done
test "$ready" -eq 1

"$PYTHON" scripts/calibrate_dspark_csd.py \
  --prompts "$PROMPTS" \
  --output "$OUT_ROOT/results/redpajama_calibration_answers.jsonl" \
  --summary "$OUT_ROOT/results/redpajama_calibration_summary.json" \
  --host 127.0.0.1 --port "$PORT" \
  --parallel 8 --max-new-tokens 512 --temperature 1.0 --top-p 1.0 \
  2>&1 | tee "$OUT_ROOT/logs/calibration_client.log"

curl --noproxy '*' -fsS -X POST "http://127.0.0.1:$PORT/save_csd_table" \
  -H 'Content-Type: application/json' \
  -d "{\"path\":\"$TABLE\",\"metadata\":{\"dataset\":\"redpajama_6domains_n1000\",\"backend\":\"mtp_eagle\",\"target_model\":\"$MODEL\",\"tree_shape\":\"3-1-4\",\"historical_cli_num_draft_tokens\":3,\"temperature\":1.0,\"top_p\":1.0,\"max_new_tokens\":512}}" \
  | tee "$OUT_ROOT/logs/export_table.json"

"$PYTHON" - "$TABLE" <<'PY' | tee "$OUT_ROOT/logs/table_validation.log"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    payload = json.load(file)
entries = payload.get("entries", [])
assert entries, "exported table is empty"
print(
    {
        "table": sys.argv[1],
        "entries": len(entries),
        "total_frequency": sum(int(entry.get("freq", 1)) for entry in entries),
    }
)
PY

echo "Calibration complete: $TABLE"
