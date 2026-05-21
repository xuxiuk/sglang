# CSD 投机解码实验说明

本分支在 SGLang EAGLE speculative decoding 上增加了 CSD 记录和回放能力。核心思路是：record 阶段只记录被拒绝 draft token 与目标 token 中满足 target logits 条件的替换关系，保存为 CSD table；replay 阶段加载 table，在验证时命中对应 pair 后，再结合当前 target logits 条件决定是否 force accept draft token。

## 实现策略

- 每个 EAGLE worker / scheduler / TPModelWorker 持有独立的 `csd_runtime`，不额外增加 TP 同步。
- record 阶段只向 GPU delta buffer 追加通过 logits gate 的 pair，不在验证热路径里做 Python 回调或重建 hash table。
- 保存 table 时通过 `/save_csd_table` flush delta buffer，并在 Python 侧聚合 pair 计数。
- replay 阶段加载静态 CSD hash table，CUDA 验证逻辑只做 table lookup 和 logits 条件检查。
- 当前实验主要使用静态 table replay；动态 table 热切换暂未作为外部 API 暴露。

## 主要参数

- `--speculative-csd`
  - 开启 CSD runtime 和验证端 CSD 逻辑。
- `--speculative-csd-dynamic-update`
  - 开启动态记录 rejected pair，用于 calibration / record 模式。
- `--speculative-csd-force-accept-disabled`
  - 禁止 force accept，只记录 pair；record 模式建议打开，避免 calibration 过程改变输出。
- `--speculative-csd-table-path`
  - replay 模式加载的 CSD table 路径。
- `--speculative-csd-freq-threshold`
  - 加载 table 时的 pair 频次过滤阈值。
- `--speculative-csd-prob-ratio`
  - 加载 table 时的概率比例过滤阈值。
- `--speculative-csd-delta-capacity`
  - record 模式 GPU delta buffer 容量；长 benchmark 建议调大，避免 pair 被截断。

## 常用模式

### Vanilla EAGLE

完全关闭 CSD：

```bash
sglang serve ... \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 3 \
  --speculative-num-draft-tokens 15
```

### Record

记录通过 target logits 条件的 CSD pair，不 force accept：

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

### Replay

加载 CSD table 并启用 force accept：

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

## 编译说明

这里没有源码目录级别的自动链接。SGLang Python 侧依赖的是当前 Python 环境里安装的 `sglang-kernel` 包，运行时通过 `import sgl_kernel` 调用已安装 wheel 里的 CUDA 扩展。

链接关系发生在这几处：

- `python/pyproject.toml` 里声明依赖 `sglang-kernel==0.4.1`。
- Python 验证逻辑在 `python/sglang/srt/speculative/eagle_utils.py` 中执行 `from sgl_kernel import verify_tree_greedy`。
- `sgl-kernel/python/sgl_kernel/__init__.py` 会加载已安装包里的 `common_ops` 扩展。
- `sgl-kernel/CMakeLists.txt` 把 speculative CUDA 源文件编进 `common_ops`，并安装到 `sgl_kernel/sm90/` 或 `sgl_kernel/sm100/`。

所以，如果修改了 `sgl-kernel/csrc/speculative/*.cu` 或 `*.cuh`，只运行 `pip install -e python` 不会重新编译 CUDA kernel。需要在同一个 conda / Python 环境里重新构建并安装本地 `sglang-kernel`：

```bash
cd sgl-kernel
make build
cd ..
```

`make build` 会自动完成两步：先 build 本地 wheel，再执行 `pip3 install dist/*whl --force-reinstall --no-deps`，把刚编译出的 `sglang-kernel` 安装到当前环境。

机器资源紧张时可以限制编译并行度：

```bash
cd sgl-kernel
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
cd ..
```

可以用下面命令确认实际加载的是当前环境中的 `sgl_kernel`：

```bash
python - <<'PY'
import sgl_kernel
print(sgl_kernel.__file__)
print(sgl_kernel.__version__)
PY
```

只修改 Python benchmark、server args 或 shell 脚本时，一般只需要：

```bash
pip install -e python --no-deps
```

## Benchmark

当前主要 benchmark 说明见：

```text
benchmark/gsm8k/README_CSD.md
```

常用流程：

1. 使用 `benchmark/redpajama/run_csd_calibration.sh` 在 RedPajama 上生成 CSD table。
2. 使用 `benchmark/gsm8k/run_csd_experiment.sh` 或 `benchmark/gsm8k/run_csd_no_tree_experiment.sh` 在 GSM8K 上 replay。
3. 使用 `benchmark/gsm8k/plot_csd_results.ipynb` 重新计算正确投机成功率并生成展示图表。
