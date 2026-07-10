#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/root/sglang}
SESSION=${SESSION:-csd_ood_v2_bfcl_sql_20260705}
LOG=${LOG:-"${REPO_ROOT}/benchmark/csd/runs/ood_v2_bfcl_sql_monitor_20260705.log"}

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${LOG}")"

while true; do
  {
    echo "============================================================"
    date
    echo "session:"
    tmux ls 2>/dev/null | rg "${SESSION}|csd_batch" || true
    echo
    echo "processes:"
    ps -ef | rg 'run_longbench_v2_csd_515|run_bfcl_csd_515|run_sql_csd_515|sglang.launch_server|30062|30063|30064' | rg -v rg || true
    echo
    echo "result files:"
    find benchmark/csd/runs -maxdepth 5 -type f \( \
      -path '*20260705_longbench_v2_domain_q20_filtered/results/*.jsonl' -o \
      -path '*20260705_bfcl_single_turn_all_cached/results/*.jsonl' -o \
      -path '*20260705_sql_create_context_q500/results/*.jsonl' \
    \) -printf '%p %s bytes\n' | sort || true
    echo
    echo "result row counts:"
    python - <<'PY' || true
import json
from pathlib import Path
paths = [
    Path('/root/sglang/benchmark/csd/runs/longbench_v2_csd_515/20260705_longbench_v2_domain_q20_filtered/results/longbench_v2_csd_515.jsonl'),
    Path('/root/sglang/benchmark/csd/runs/bfcl_csd_515/20260705_bfcl_single_turn_all_cached/results/bfcl_csd_515.jsonl'),
    Path('/root/sglang/benchmark/csd/runs/sql_csd_515/20260705_sql_create_context_q500/results/sql_csd_515.jsonl'),
]
for path in paths:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    counts = {}
    for row in rows:
        model_id = row.get('model_id') or row.get('other', {}).get('model_id', '')
        method = model_id.replace('Qwen3.5-35B-A3B_', '')
        if row.get('throughput') is not None:
            counts[method] = counts.get(method, 0) + 1
    print(path, len(rows), counts)
PY
    echo
    echo "recent errors:"
    find benchmark/csd/runs -maxdepth 6 -type f \( \
      -path '*20260705_longbench_v2_domain_q20_filtered/logs/*.log' -o \
      -path '*20260705_bfcl_single_turn_all_cached/logs/*.log' -o \
      -path '*20260705_sql_create_context_q500/logs/*.log' \
    \) -print0 | xargs -0 -r rg -n -i 'traceback|error:|failed|exception|HTTPError|CUDA out of memory|longer than the model' | tail -80 || true
    echo
    echo "pane tail:"
    tmux capture-pane -pt "${SESSION}:0" -S -40 2>/dev/null || true
  } >>"${LOG}" 2>&1

  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session ${SESSION} ended at $(date)" >>"${LOG}"
    break
  fi
  sleep 3600
done
