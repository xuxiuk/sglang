#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=${RUN_ROOT:-${HERE}/runs/full_pipeline_20260813_015303}
STRICT_RUN=${STRICT_RUN:-$(find "${RUN_ROOT}" -maxdepth 1 -type d -name 'strict_rejudge_*' | sort | tail -n 1)}
CALIBRATION_TABLE=${CALIBRATION_TABLE:-/root/sglang-dspark-csd/runs/mtp_csd/qwen35b_mtp314_final/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps3_topk1_draft3_temp1.0_ratio0.01.json}
OUTPUT_DIR=${OUTPUT_DIR:-${STRICT_RUN}/frequency_analysis}
PYTHON=${PYTHON:-/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python}

if [[ -z "${STRICT_RUN}" ]]; then
  echo "No strict_rejudge_* directory found below ${RUN_ROOT}" >&2
  exit 1
fi

judge_files=()
for task in lcb_v6 aime25 olympiad_math_en; do
  file="${STRICT_RUN}/${task}/judge/judged_strict.jsonl"
  test -s "${file}" || {
    echo "Strict judgment is not complete: ${file}" >&2
    exit 1
  }
  judge_files+=("${file}")
done

"${PYTHON}" "${HERE}/analyze_misrejection_frequency.py" \
  --judge "${judge_files[@]}" \
  --calibration-table "${CALIBRATION_TABLE}" \
  --freq-threshold 6 \
  --positive-labels PROVEN_EQUIVALENT DRAFT_BETTER \
  --output-dir "${OUTPUT_DIR}"
