# CSD Benchmark

`benchmark/csd/` 是本分支 CSD speculative decoding 实验的集中入口。这里把 RedPajama calibration、GSM8K replay、lm-eval、LightEval 和结果输出统一放在一个目录下，避免 CSD 相关脚本继续散落在 `benchmark/gsm8k/` 或 `benchmark/` 顶层。

## 功能和作用

本分支在 SGLang EAGLE speculative decoding 上增加了 CSD（candidate substitution decoding）记录、回放和在线更新能力。

核心思路是：

- record / calibration 阶段：记录被拒绝的 draft token 与 target token 的替换关系，保存为 CSD table。
- replay 阶段：加载 CSD table，在验证时如果命中 `(draft_token, target_token)` pair，再结合 target logits 条件决定是否 force accept draft token。
- online update 阶段：在 replay 的同时继续收集新的 rejected pair，达到阈值后异步重建 CSD hash table，并在下一次 verify 边界切换到新表。

这个目录主要服务于几类实验：

- 用 RedPajama 生成跨领域 CSD calibration table。
- 在 GSM8K 上比较 baseline、vanilla EAGLE 和 CSD EAGLE。
- 用 lm-eval native runner 导出 speculative / CSD 指标。
- 用 LightEval in-process SGLang backend 跑代码、数学等任务，并导出和 lm-eval 对齐的指标。

## 目录结构

```text
benchmark/csd/
├── README.md                         # 当前主说明
├── eval/
│   ├── lm_eval_sglang_native.py
│   ├── run_lighteval_csd_experiment.sh
│   ├── run_lighteval_sglang_native.py
│   ├── run_lm_eval_csd_experiment.sh
│   └── run_lm_eval_sglang_native.py
├── gsm8k/
│   ├── README.md
│   ├── bench_other.py
│   ├── bench_sglang.py
│   ├── bench_sglang_eagle.py
│   ├── run_csd_experiment.sh
│   └── run_csd_no_tree_experiment.sh
├── redpajama/
│   ├── bench_redpajama_csd.py
│   └── run_csd_calibration.sh
└── runs/                             # 默认输出目录，运行脚本时自动创建
```

默认输出位置：

```text
benchmark/csd/runs/redpajama/      # RedPajama calibration 输出，也是默认 CSD_TABLE_PATH 来源
benchmark/csd/runs/gsm8k/          # GSM8K tree EAGLE CSD 输出
benchmark/csd/runs/gsm8k_no_tree/  # GSM8K no-tree / top-k 1 EAGLE CSD 输出
benchmark/csd/runs/lm_eval/        # lm-eval CSD 输出
benchmark/csd/runs/lighteval/      # LightEval CSD 输出
benchmark/csd/runs/profiles/       # profiler 输出
```

这些路径可以通过 `OUT_DIR`、`CSD_TABLE_PATH`、`SGLANG_TORCH_PROFILER_DIR` 等环境变量覆盖。

## 主要代码改动

### SGLang runtime

- `python/sglang/srt/speculative/csd_runtime.py`
  - 管理 CSD table、GPU delta buffer、metrics、保存和加载。
  - 增加 CPU hash table payload，用于把 hash table 构建拆成 CPU 构表和 GPU materialize 两步。
  - 增加线程池异步 rebuild：后台线程只构建 CPU payload，不直接操作 CUDA tensor。
  - 增加 `maybe_start_async_rebuild()` 和 `maybe_apply_async_rebuild()`。

- `python/sglang/srt/server_args.py`
  - 增加 CSD 相关参数。
  - 增加 `--speculative-csd-rebuild-threshold`。
  - `--speculative-csd-delta-capacity` 默认值改为按模式后处理：record 模式默认大 buffer，online update 模式默认较小 buffer。

- `python/sglang/srt/speculative/eagle_worker.py`
  - 在 EAGLE v1 verify 前应用已完成的异步 rebuild。
  - 在 verify 后按阈值尝试启动下一轮异步 rebuild。

- `python/sglang/srt/speculative/eagle_worker_v2.py`
  - 在 EAGLE v2 sample 前后接入同样的异步 rebuild 时序。

- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
  - 把 CSD counters 写入 response metadata，benchmark runner 可以读取。

### CUDA kernel

- `sgl-kernel/csrc/speculative/eagle_utils.cu`
- `sgl-kernel/csrc/speculative/speculative_sampling.cuh`

CSD 在 verify kernel 中完成：

- 对 rejected draft pair 执行 delta append。
- 对 CSD table 做 hash lookup。
- table 命中后检查 logits 条件。
- 满足条件时 force accept draft token。
- 更新 `csd_lookup_hit_ct`、`csd_forced_accept_ct`、`csd_delta_pair_ct`。

### Benchmark / eval

- `benchmark/csd/gsm8k/bench_sglang_eagle.py`
  - 记录 GSM8K accuracy、latency、throughput、accept length 和 CSD counters。

- `benchmark/csd/redpajama/bench_redpajama_csd.py`
  - 从 RedPajama 多领域文本生成 calibration table。

- `benchmark/csd/eval/run_lm_eval_sglang_native.py`
  - 使用 SGLang `/generate` 路径运行 lm-eval 风格任务。
  - 导出 speculative / CSD metrics JSON 和 compact JSONL。

- `benchmark/csd/eval/run_lighteval_sglang_native.py`
  - 使用 LightEval 的 in-process SGLang backend。
  - 直接构造 `sglang.Engine(...)`，不走额外 HTTP `/generate` backend。
  - 导出和 lm-eval 对齐的 speculative / CSD 指标。

## 主要参数

### 基础 speculative 参数

- `--speculative-algorithm EAGLE`
  - 开启 EAGLE speculative decoding。

- `--speculative-num-steps`
  - EAGLE sequential draft depth。

- `--speculative-eagle-topk`
  - EAGLE 每步 draft top-k。

- `--speculative-num-draft-tokens`
  - tree EAGLE 的 draft candidate 节点数。

### CSD 参数

- `--speculative-csd`
  - 开启 CSD runtime 和 verify 端 CSD 逻辑。
  - Python `ServerArgs` 字段名是 `speculative_csd_enabled`。

- `--speculative-csd-table-path`
  - replay / online update 模式加载的 CSD table 路径。

- `--speculative-csd-dynamic-update`
  - 开启 rejected pair 记录。
  - record 模式用它生成 table。
  - online update 模式用它在 replay 时继续收集新 pair。

- `--speculative-csd-force-accept-disabled`
  - 禁止 force accept，只记录 pair。
  - record / calibration 模式建议开启，避免 calibration 改变模型输出。

- `--speculative-csd-freq-threshold`
  - 加载或 rebuild table 时的 pair 频次过滤阈值。
  - 只有累计频次达到该阈值的 pair 会进入 hash table。

- `--speculative-csd-prob-ratio`
  - CSD force accept 的 logits 条件参数。

- `--speculative-csd-delta-save-path`
  - 动态收集的 delta pair 保存路径。

- `--speculative-csd-delta-capacity`
  - GPU delta buffer 容量。
  - 如果用户不显式传入，会在 `ServerArgs.__post_init__()` 中按模式设置：
    - record / calibration：默认 `1 << 20`。
    - online update：默认 `2 * speculative_csd_rebuild_threshold`。

- `--speculative-csd-rebuild-threshold`
  - 在线更新模式下，触发异步 rebuild 的最小 buffered pair 数。
  - 默认 `4096`。
  - 设置为 `<= 0` 可以关闭 online rebuild。

### 并发相关参数

- `--max-running-requests`
  - SGLang scheduler 同时运行请求数。
  - speculative decoding 下如果不显式设置，SGLang 会默认重置为 `48`。
  - LightEval in-process SGLang backend 也支持传这个参数。

## 运行模式

### 1. Baseline

不开 speculative decoding：

```bash
sglang serve ...
```

用于对比普通 decoding 的准确率和吞吐。

### 2. Vanilla EAGLE

开启 EAGLE，但不开 CSD：

```bash
sglang serve ... \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 3 \
  --speculative-num-draft-tokens 15
```

### 3. Record / calibration

记录 rejected pair，不 force accept：

```bash
sglang serve ... \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 3 \
  --speculative-num-draft-tokens 15 \
  --speculative-csd \
  --speculative-csd-dynamic-update \
  --speculative-csd-force-accept-disabled \
  --speculative-csd-delta-capacity 16777216
```

典型用途是 RedPajama calibration 或 GSM8K 本地 record。这个模式下没有 `--speculative-csd-table-path`，因此不会在线 rebuild，只会记录 delta。

### 4. Static replay

加载已有 CSD table 并启用 force accept：

```bash
sglang serve ... \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 3 \
  --speculative-num-draft-tokens 15 \
  --speculative-csd \
  --speculative-csd-table-path /path/to/csd_table.json \
  --speculative-csd-freq-threshold 3 \
  --speculative-csd-prob-ratio 0.01
```

### 5. Online update replay

加载已有 table，同时继续记录新 pair 并异步更新 table：

```bash
sglang serve ... \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 3 \
  --speculative-num-draft-tokens 15 \
  --speculative-csd \
  --speculative-csd-table-path /path/to/csd_table.json \
  --speculative-csd-dynamic-update \
  --speculative-csd-freq-threshold 3 \
  --speculative-csd-prob-ratio 0.01 \
  --speculative-csd-rebuild-threshold 4096
```

这个模式下如果不显式传 `--speculative-csd-delta-capacity`，默认会设置为：

```text
2 * speculative_csd_rebuild_threshold
```

## 在线更新实现逻辑

在线更新只在 replay + dynamic update 模式启用：

```text
speculative_csd_dynamic_update = true
speculative_csd_table_path != None
```

纯 record / calibration 模式没有 `speculative_csd_table_path`，因此只记录 delta，不会在线 rebuild。

在线更新的高层时序是：

```text
verify 前：如果上一轮 rebuild 已完成，就切换到新 CSD table
verify 中：当前 table 做 lookup / force accept，CUDA kernel 继续收集 rejected pair
verify 后：如果 delta buffer 达到 rebuild_threshold，就 flush delta 并启动后台 CPU 构表
```

这里的“异步”指 hash table payload 在后台线程构建，避免每次触发 rebuild 时都把完整 CPU 构表同步塞进 verify 热路径。后台线程不直接创建 CUDA tensor；GPU tensor 创建和 table 替换仍然在 verify 边界由主线程完成。

### Delta buffer 计数语义

CUDA 侧 `delta_counter` 是 append 尝试次数，可能超过 `delta_capacity`。Python 侧实际可读取 pair 数是：

```python
min(delta_counter, delta_capacity)
```

因此在线 rebuild 触发判断基于实际可读取 pair 数，而不是单纯 counter。

## Hash table 构建逻辑

CSD table 使用开放寻址 hash table。建表逻辑由以下参数控制：

- `CSD_DEFAULT_LOAD_FACTOR = 0.5`
- `CSD_DEFAULT_MAX_PROBE = 16`

建表流程：

```text
filtered keys -> next_power_of_two(len(keys) / load_factor) -> linear probing insert
```

如果在当前 capacity 和 max_probe 下插入失败，会把 capacity 翻倍后重试。

注意：`delta_capacity` 不直接影响 hash probe 次数。它只影响一次 flush 能从 GPU 收集多少 rejected pair。真正影响 hash table probe 的是：

- key 数量
- load factor
- max probe
- hash 分布

## 常用入口

RedPajama CSD table calibration：

```bash
bash benchmark/csd/redpajama/run_csd_calibration.sh
```

GSM8K tree EAGLE CSD experiment：

```bash
bash benchmark/csd/gsm8k/run_csd_experiment.sh
```

GSM8K no-tree / top-k 1 EAGLE CSD experiment：

```bash
bash benchmark/csd/gsm8k/run_csd_no_tree_experiment.sh
```

lm-eval CSD experiment：

```bash
bash benchmark/csd/eval/run_lm_eval_csd_experiment.sh
```

LightEval CSD experiment：

```bash
LIGHTEVAL_PYTHON=/home/zhouxuwen/miniconda3/envs/lighteval-sglang/bin/python \
  bash benchmark/csd/eval/run_lighteval_csd_experiment.sh
```

## 关键指标

- `accuracy`
  - 任务准确率。

- `latency`
  - 端到端耗时。

- `throughput`
  - 输出 token/s。

- `request_throughput`
  - request/s。

- `accept_length`
  - 平均每次 verification 产生的输出长度。

- `aggregate_accept_length`
  - 用全局累计 token 和 verify 次数计算出的 accept length。

- `spec_success_rate`
  - speculative 成功率，常用计算：

```python
spec_success_rate = (accept_length - 1) / speculative_num_steps
```

注意：tree EAGLE 里 `speculative_num_draft_tokens` 是候选节点总数，不是 sequential draft length，所以不要拿它当分母。

- `spec_output_token_saved_ratio`
  - 输出 token saved ratio，和 speculative success rate 不是同一个指标。

- `csd_lookup_hit_ct`
  - CSD table lookup 命中次数。

- `csd_forced_accept_ct`
  - CSD force accept 次数。

- `csd_delta_pair_ct`
  - dynamic update / record 阶段成功写入 delta buffer 的 pair 数。

## 编译说明

SGLang Python 侧依赖当前 Python 环境里安装的 `sglang-kernel` 包。运行时通过 `import sgl_kernel` 调用已安装 wheel 里的 CUDA 扩展。

如果修改了：

```text
sgl-kernel/csrc/speculative/*.cu
sgl-kernel/csrc/speculative/*.cuh
```

需要重新构建并安装 `sglang-kernel`：

```bash
cd sgl-kernel
make build
cd ..
pip install -e python --no-deps
```

只修改 Python benchmark、server args 或 shell 脚本时，一般只需要：

```bash
pip install -e python --no-deps
```

可以用下面命令确认实际加载的是当前环境中的 `sgl_kernel`：

```bash
python - <<'PY'
import sgl_kernel
print(sgl_kernel.__file__)
print(sgl_kernel.__version__)
PY
```

## 备注

- 每个 EAGLE worker / scheduler / TPModelWorker 持有自己的 `csd_runtime`，不额外增加 TP 同步。
- record 阶段只向 GPU delta buffer 追加 pair，不在 verify 热路径里做 Python 回调。
- online update 阶段的 rebuild 使用后台线程构建 CPU hash payload；CUDA tensor 创建和 table 替换仍然在 verify 边界完成。
- `benchmark/csd/` 中保留的是集中后的脚本和 README；原始文件暂未删除，避免破坏已有命令、历史结果或 notebook 引用。
