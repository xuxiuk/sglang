#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export JUDGE_SCRIPT="${HERE}/judge_replayed_branches_v4.py"
export PROMPT_NAME=V4
export OUTPUT_BASENAME=judged_v4.jsonl
export SUMMARY_JSON=summary_v4.json
export SUMMARY_MD=summary_v4.md
export MAX_RETRIES=${MAX_RETRIES:-3}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${HERE}/runs/full_pipeline_20260813_015303/v4_judge_$(date +%Y%m%d_%H%M%S)}

exec bash "${HERE}/run_v3_judge_all.sh"
