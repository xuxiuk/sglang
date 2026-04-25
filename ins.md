# GSM8K CSD experiment commands

Assumptions:

- Run from repo root: `/home/zhouxuwen/sglang`.
- Use GSM8K 1500 examples, client concurrency `--parallel 1`, temperature `1.0`.
- `--parallel 1` means the benchmark keeps at most one unfinished request in flight, so server-side request batch size is effectively 1 for this run.
- Use MTP-style draft, so benchmark metadata records `--draft-model-name mtp`.
- Record mode collects CSD pairs but disables force accept.
- CSD mode loads the recorded table and enables force accept.
- Stop the previous server before launching the next one.

Common paths:

```bash
export OUT_DIR=/home/zhouxuwen/sglang/benchmark/gsm8k/csd_runs
export DATA_PATH=/home/zhouxuwen/sglang/benchmark/gsm8k/test.jsonl
export MODEL_PATH=/home/shared/models/Qwen/Qwen3.5-35B-A3B
export CSD_TABLE_PATH=${OUT_DIR}/csd_gsm8k_n1500_Qwen3.5-35B-A3B_mtp_temp1_top_p1_record.json
mkdir -p ${OUT_DIR}
```

If `${DATA_PATH}` does not exist, either place GSM8K test jsonl there first, or let the benchmark download it by omitting `--data-path`.

## 1. Launch record-mode server

```bash
CUDA_VISIBLE_DEVICES=6,7 \
SGLANG_TORCH_PROFILER_DIR=/home/zhouxuwen/sglang/profiles \
sglang serve \
  --model-path ${MODEL_PATH} \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 3 \
  --speculative-num-draft-tokens 15 \
  --mem-fraction-static 0.7 \
  --mamba-scheduler-strategy extra_buffer \
  --watchdog-timeout 3000 \
  --log-level warning \
  --port 30000 \
  --speculative-csd \
  --speculative-csd-dynamic-update \
  --speculative-csd-force-accept-disabled
```

## 2. Run record-mode GSM8K benchmark

This writes decoded answers, raw rows, aggregate result JSONL, and the CSD table.

```bash
python benchmark/gsm8k/bench_sglang_eagle.py \
  --num-questions 1500 \
  --num-shots 5 \
  --parallel 8 \
  --max-new-tokens 512 \
  --temperature 1.0 \
  --top-p 1.0 \
  --host 127.0.0.1 \
  --port 30000 \
  --backend srt \
  --answer-file ${OUT_DIR}/record_gsm8k_n1500_temp1_answer.txt \
  --raw-result-file ${OUT_DIR}/record_gsm8k_n1500_temp1_raw.jsonl \
  --result-file ${OUT_DIR}/result.jsonl \
  --dataset-name gsm8k \
  --model-name ${MODEL_PATH} \
  --draft-model-name mtp \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 3 \
  --speculative-num-draft-tokens 15 \
  --run-tag record \
  --csd-enabled \
  --csd-dynamic-update \
  --csd-force-accept-disabled \
  --csd-freq-threshold 6 \
  --csd-prob-ratio 0.01 \
  --csd-log-result \
  --csd-save-table-path ${CSD_TABLE_PATH}
```

Check that the table was saved and has entries:

```bash
python - <<'PY'
import json, os
path = os.environ['CSD_TABLE_PATH']
payload = json.load(open(path))
print('table:', path)
print('entries:', len(payload.get('entries', [])))
print('metadata:', payload.get('metadata', {}))
PY
```

## 3. Launch CSD-enabled server

```bash
CUDA_VISIBLE_DEVICES=6,7 \
SGLANG_TORCH_PROFILER_DIR=/home/zhouxuwen/sglang/profiles \
sglang serve \
  --model-path ${MODEL_PATH} \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 12 \
  --mem-fraction-static 0.7 \
  --mamba-scheduler-strategy extra_buffer \
  --watchdog-timeout 3000 \
  --log-level warning \
  --port 30000 \
  --max-running-requests 1 \
  --speculative-csd \
  --speculative-csd-table-path ${CSD_TABLE_PATH} \
  --speculative-csd-freq-threshold 6 \
  --speculative-csd-prob-ratio 0.01
```

## 4. Run CSD-enabled GSM8K benchmark

```bash
python benchmark/gsm8k/bench_sglang_eagle.py \
  --num-questions 1500 \
  --num-shots 5 \
  --parallel 8 \
  --max-new-tokens 512 \
  --temperature 1.0 \
  --top-p 1.0 \
  --host 127.0.0.1 \
  --port 30000 \
  --backend srt \
  --answer-file ${OUT_DIR}/csd_gsm8k_n1500_temp1_answer.txt \
  --raw-result-file ${OUT_DIR}/csd_gsm8k_n1500_temp1_raw.jsonl \
  --result-file ${OUT_DIR}/result.jsonl \
  --dataset-name gsm8k \
  --model-name ${MODEL_PATH} \
  --draft-model-name mtp \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 3 \
  --speculative-num-draft-tokens 15 \
  --run-tag csd \
  --csd-enabled \
  --csd-table-path ${CSD_TABLE_PATH} \
  --csd-freq-threshold 6 \
  --csd-prob-ratio 0.01 \
  --csd-log-result
```

## 5. Compare results

```bash
python - <<'PY'
import json, os
path = os.path.join(os.environ['OUT_DIR'], 'result.jsonl')
for line in open(path):
    row = json.loads(line)
    other = row.get('other', {})
    print({
        'run_tag': other.get('run_tag'),
        'accuracy': row.get('accuracy'),
        'invalid': row.get('invalid'),
        'throughput': row.get('throughput'),
        'accept_length': row.get('accept_length'),
        'spec_success_rate': row.get('spec_success_rate'),
        'csd_lookup_hit_ct': other.get('csd_lookup_hit_ct'),
        'csd_forced_accept_ct': other.get('csd_forced_accept_ct'),
        'csd_delta_pair_ct': other.get('csd_delta_pair_ct'),
        'parallel': other.get('parallel'),
        'temperature': other.get('temperature'),
        'speculative': other.get('speculative'),
    })
PY
```
