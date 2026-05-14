# GSM8K CSD Benchmark 说明

这个 README 说明当前实际使用的 CSD/GSM8K benchmark 流程。主要入口在 `benchmark/gsm8k/`，RedPajama calibration 入口在 `benchmark/redpajama/`。

## 目录关系

- `benchmark/gsm8k/`：当前主要运行目录，包含 GSM8K CSD replay、record/replay 脚本和结果展示 notebook。
- `benchmark/redpajama/`：当前主要 RedPajama calibration 目录，用来生成 CSD table。
- `benchmark/csd/`：之前复制出来的独立副本，更像实验快照；后续如果继续直接跑 `benchmark/gsm8k/`，README 应该优先看这里。

当前 `benchmark/csd/gsm8k` 和 `benchmark/gsm8k` 不是完全一致：核心 `bench_sglang_eagle.py`、`run_csd_experiment.sh`、`run_csd_no_tree_experiment.sh` 基本一致，但 `benchmark/csd/gsm8k` 缺少本地 record/replay 脚本和结果 notebook。

## 编译和安装

`pip install -e .` 或 `pip install -e python` 只会安装 SGLang Python 侧代码，不会自动重新编译 `sgl-kernel/` 里的 CUDA/C++ kernel。

如果改了 `sgl-kernel/csrc/speculative/*.cu`、`*.cuh` 或 C++ binding，需要先重新编译并安装 `sglang-kernel`：

```bash
cd sgl-kernel
make build
cd ..
pip install -e python
```

机器资源紧张时可以限制编译并行度：

```bash
cd sgl-kernel
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
cd ..
pip install -e python
```

只改 Python benchmark、server args 或脚本时，一般只需要：

```bash
pip install -e python
```

## 核心改动

- `python/sglang/srt/speculative/csd_runtime.py`
  - 管理 CSD table、delta buffer、保存/加载和运行时 counters。
  - 每个 EAGLE worker / scheduler / TPModelWorker 持有自己的 `csd_runtime`，不额外做 TP 同步。
- `python/sglang/srt/server_args.py`
  - 增加 CSD 参数，包括 `--speculative-csd-delta-capacity`。
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
  - 把 CSD counters 写入 response metadata，benchmark 可以读取。
- `benchmark/gsm8k/bench_sglang_eagle.py`
  - 记录 GSM8K accuracy、latency、throughput、accept length、CSD counters。
- `benchmark/redpajama/bench_redpajama_csd.py`
  - 从 RedPajama 六个领域读取文本，用于生成 calibration table。

## 运行模式

- `baseline`：不开 speculative decoding。
- `vanilla`：开 EAGLE，但完全关闭 CSD。
- `record`：开 CSD dynamic update，只记录 pair，不 force accept。
- `csd`：加载 CSD table，命中后按 logit 条件尝试 force accept。

## 关键指标

- `accuracy`：GSM8K 准确率。
- `throughput`：输出 token/s。
- `accept_length`：平均每次 verification 产生的输出长度。
- `spec_success_rate`：正确投机成功率，使用：

```python
spec_success_rate = (accept_length - 1) / speculative_num_steps
```

注意：tree EAGLE 里 `speculative_num_draft_tokens` 是候选节点总数，不是 sequential draft length，所以不要拿它当分母。

- `spec_output_token_saved_ratio`：输出 token saved ratio，和投机成功率不是同一个指标。
- `csd_lookup_hit_ct`：CSD table 命中次数。
- `csd_forced_accept_ct`：CSD force accept 次数。
- `csd_delta_pair_ct`：record 阶段收集到的 pair 数量。

## RedPajama 生成 CSD Table

```bash
CUDA_DEVICES=4,5 \
TP_SIZE=2 \
SAMPLES_PER_DOMAIN=1500 \
PARALLEL=8 \
TEMPERATURE=1.0 \
SPEC_NUM_STEPS=5 \
SPEC_TOPK=3 \
SPEC_DRAFT_TOKENS=15 \
CSD_FREQ_THRESHOLD=3 \
CSD_PROB_RATIO=0.01 \
CSD_DELTA_CAPACITY=16777216 \
bash benchmark/csd/redpajama/run_csd_calibration.sh
```

默认输出在：

```text
benchmark/csd/runs/redpajama/
```

RedPajama 使用六个领域：`arxiv`、`c4`、`common_crawl`、`github`、`stackexchange`、`wikipedia`。

## GSM8K 使用 RedPajama Table 测试

Tree EAGLE：

```bash
CSD_TABLE_PATH=/path/to/csd_table_redpajama.json \
bash benchmark/csd/gsm8k/run_csd_experiment.sh
```

No-tree / top-k 1 EAGLE：

```bash
CSD_TABLE_PATH=/path/to/csd_table_redpajama.json \
bash benchmark/csd/gsm8k/run_csd_no_tree_experiment.sh
```

这两个脚本都会跑：baseline、vanilla EAGLE、CSD replay。

## GSM8K 本地 Record/Replay

如果想直接用 GSM8K 自己生成 table，再回放测试：

```bash
bash benchmark/gsm8k/run_csd_record_experiment.sh
bash benchmark/gsm8k/run_csd_no_tree_record_experiment.sh
```

## 结果展示

```bash
jupyter notebook benchmark/gsm8k/plot_csd_results.ipynb
```

旧版 notebook 的输入路径仍指向原始 GSM8K 输出目录。集中目录下新脚本默认输出到 `benchmark/csd/runs/gsm8k/result.jsonl`，如需继续使用旧 notebook，请相应调整输入路径。输出建议保存到：

```text
benchmark/csd/runs/gsm8k/plots/
```
