#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=/root/sglang-dspark-csd/runs/mtp_csd/qwen35b_mtp314_final
PYTHON=/root/miniconda3/envs/sglang-dspark-csd-cu128/bin/python

for dataset in apps taco; do
  result_root="$RUN_DIR/results"
  answer_count=$(find "$result_root/$dataset/answers" -maxdepth 1 \
    -name '*_model_outputs.json' -type f -size +0c 2>/dev/null | wc -l)
  if [[ "$answer_count" -ne 4 ]]; then
    echo "Expected four completed $dataset answer files, got $answer_count" >&2
    exit 1
  fi
  mkdir -p "$result_root/$dataset/accuracy"
  "$PYTHON" "$RUN_DIR/evaluate_code_ood_pass4.py" \
    --dataset "$dataset" \
    --run-root "$result_root" \
    --prepared-data "$RUN_DIR/vendor/assets/data/${dataset}_$([[ $dataset == apps ]] && echo 2000 || echo 1000)_code_ood_alpaca.json" \
    --limit 1000 --n 4 --workers 16 \
    2>&1 | tee "$result_root/${dataset}/accuracy/evaluator.log"
done
