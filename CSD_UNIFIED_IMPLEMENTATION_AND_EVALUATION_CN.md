# CSD 统一接入与评测说明：MTP、DFlash 与 DSpark

本文说明当前仓库如何把 CSD 接入三种投机解码后端，以及启动、校准和评测时需要关注的参数。这里有两个不同层次：

- 三种投机后端：MTP/EAGLE、DFlash、DSpark；
- 三种 CSD 模式：plain、dynamic、dynamic + entropy gate。

CSD 不负责生成 draft token。三个后端仍使用各自原生 proposer，CSD 只增强 target verifier 的接受阶段。

## 1. 实现改动与 Kernel 构建

### 1.1 代码改动总览

本实现是在上游已有的 MTP/EAGLE、DFlash 和 DSpark 上增加一套公共 CSD runtime，
并扩展它们的 accept/rejection verifier。DSpark 本身及其 compact/SPS 路径来自当前固定的
上游版本，不是本项目重新实现；下表只说明本项目为接入 CSD 新增或扩展的代码。

#### 公共控制面与 runtime

| 文件 | 类型 | 本项目中的职责 |
| --- | --- | --- |
| `python/sglang/srt/server_args.py` | 扩展上游文件 | 注册 CSD 公共启动参数和默认值 |
| `python/sglang/srt/arg_groups/speculative_hook.py` | 扩展上游文件 | 校验 CUDA、后端、table path、ratio、entropy 和 dynamic 参数组合，并补全 delta capacity |
| `python/sglang/srt/speculative/csd_runtime.py` | **新增核心文件** | 加载/导出 pair-frequency store，构建 GPU hash table，管理 metrics、delta buffer、跨 DP 聚合和异步 rebuild 生命周期 |

#### 三种后端的接入点

| 文件 | 类型 | 本项目中的职责 |
| --- | --- | --- |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | 扩展上游文件 | 为 MTP/EAGLE 创建公共 `CSDRuntime`，在 decode 边界驱动 rebuild，并提供 table 导出 |
| `python/sglang/srt/speculative/eagle_utils.py` | 扩展上游文件 | 将 MTP 的 greedy/sampling verifier 参数接入公共 CSD kernel |
| `python/sglang/srt/speculative/dflash_worker_v2.py` | 扩展上游文件 | 为 DFlash 创建 runtime、驱动 rebuild、校验/导出 DFlash table metadata |
| `python/sglang/srt/speculative/dflash_utils.py` | 扩展上游文件 | 将 DFlash 线性 block 转成公共 verifier 所需布局，并接入 greedy/sampling CSD 路径 |
| `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` | 扩展上游文件 | 为 DSpark 创建 runtime，在 DP/compact decode 中驱动 rebuild 和导出 |
| `python/sglang/srt/speculative/dspark_components/dspark_accept.py` | 扩展上游文件 | 把 runtime 传入 DSpark accept 层，并避免 mixed greedy/sampling batch 重复记录同一 observation |
| `python/sglang/srt/speculative/dspark_components/kernels/accept_greedy.py` | 扩展上游文件 | DSpark greedy 开启 CSD 时切换到公共 CUDA greedy verifier |
| `python/sglang/srt/speculative/dspark_components/kernels/accept_sampling.py` | 扩展上游文件 | DSpark sampling 开启 CSD 时从原生 chain Triton verifier 切到支持 CSD 与 rejection sampling 的 CUDA verifier |
| `python/sglang/srt/speculative/dspark_components/dspark_csd.py` | **新增适配文件** | DSpark 侧导出公共 `csd_kernel_kwargs`，避免复制另一套 CSD 语义 |

#### Kernel、operator 与构建

| 文件 | 类型 | 本项目中的职责 |
| --- | --- | --- |
| `sgl-kernel/csrc/speculative/csd_rebuild.cpp` | **新增核心文件** | `CSDTableBuilder` 的 CPU 频次累积、开放寻址 hash table 的构建/rebuild，以及 builder/operator 注册 |
| `sgl-kernel/csrc/speculative/speculative_sampling.cu/.cuh` | 扩展上游文件 | sampling verifier 中融合 hash lookup、probability ratio、entropy gate、delta append 和 rejection-sampling 支持 |
| `sgl-kernel/csrc/speculative/eagle_utils.cu` | 扩展上游文件 | greedy tree verifier 中融合 CSD lookup、ratio gate、force accept、delta append 和 counters |
| `sgl-kernel/include/sgl_kernel_ops.h` | 扩展上游文件 | 声明扩展后的 C++ verifier 函数签名 |
| `sgl-kernel/csrc/common_extension.cc` | 扩展上游文件 | 扩展两个 verifier 的 PyTorch operator schema 并绑定 CUDA 实现 |
| `sgl-kernel/python/sgl_kernel/speculative.py` | 扩展上游文件 | 暴露 Python 可调用的 speculative/CSD operator 接口 |
| `sgl-kernel/CMakeLists.txt` | 扩展上游文件 | 将新增的 `csd_rebuild.cpp` 编入 `sgl-kernel` |

#### HTTP 导出、指标与控制消息

| 文件 | 类型 | 本项目中的职责 |
| --- | --- | --- |
| `python/sglang/srt/managers/io_struct.py` | 扩展上游文件 | 定义 save-table 请求和逐 DP shard 返回结构 |
| `python/sglang/srt/managers/tokenizer_control_mixin.py` | 扩展上游文件 | 将 save-table 控制请求广播到各 scheduler/DP rank |
| `python/sglang/srt/managers/scheduler.py` | 扩展上游文件 | 处理 table 导出、暴露 CSD counters，并给出精确 speculative 累计计数 |
| `python/sglang/srt/entrypoints/http_server.py` | 扩展上游文件 | 新增 `/save_csd_table` HTTP 接口 |

#### 测试与实验工具

Kernel 单元测试扩展在
`sgl-kernel/tests/speculative/test_eagle_utils.py`、
`test_speculative_sampling.py` 和新增的 `test_dspark_csd.py`；参数组合测试位于
`test/registered/unit/server_args/test_server_args.py` 与
`test/registered/unit/speculative/`。这些测试覆盖 greedy/sampling lookup、force accept、
dynamic observation、entropy/rejection sampling 和非法启动参数。

`scripts/run_dflash_csd.sh`、`scripts/run_dspark_csd.sh`、
`scripts/merge_csd_tables.py`、`scripts/eval_dspark_long.py` 和
`scripts/summarize_dspark_task_metrics.py` 是新增的校准/评测工具，不参与 server 的在线执行；
MTP、DFlash、DSpark 的正式脚本与输出索引见第 8、9 节。

### 1.2 环境与 Kernel 构建

除 kernel 外，Python/SGLang 主环境按 SGLang 官方教程正常构建即可，本实现没有额外的 CSD Python 环境安装步骤。

CSD 不是纯 Python 功能。本仓库的以下部分编译进 `sgl-kernel`：

- `speculative_sampling.cu/.cuh`：table lookup、probability gate、entropy gate、delta append；
- `eagle_utils.cu`：MTP/EAGLE verifier 的 CSD 路径；
- `csd_rebuild.cpp`：`CSDTableBuilder` 的本地频次累积、hash table rebuild 和 builder/operator 注册；
- `common_extension.cc`：扩展 verifier 的 PyTorch operator schema；

CSD 源文件已经加入 `sgl-kernel/CMakeLists.txt`，因此**没有 CSD 专用的 CMake 选项、额外 patch 步骤或单独 build target**。但本仓库修改了 AOT C++/CUDA kernel，首次使用这份源码，或更新了 `sgl-kernel` 中的 CSD 实现后，必须在用于运行服务的同一个 Python 环境中重新完整构建并安装 kernel：

```bash
conda activate sglang-dspark-csd-cu128
cd /root/sglang-dspark-csd/sgl-kernel
make rebuild MAX_JOBS=96 CMAKE_BUILD_PARALLEL_LEVEL=96
```

`make rebuild` 会清理旧产物、重新构建 wheel，并安装到当前环境。构建完成后重启服务即可。

## 2. CSD 的公共流程

### 2.1 Frequency store 到 GPU hash table

Calibration 导出的 JSON/JSONL 保存的是 CPU 侧的 pair-frequency store，即
`(draft_token, residual_token) -> frequency`，它本身不会被 verifier 直接查询。服务启动时先执行：

1. 从 `--speculative-csd-table-path` 读取全部 pair 及累计频次；未提供路径时创建空 store；
2. 用 `frequency >= freq_threshold` 筛出 active pair；
3. 将每个 active pair 打包成一个 `int64` key；
4. `csd_build_hash_table_cpu` 在 CPU 上构建只保存 key 的开放寻址 hash table；
5. 将 hash table 的 `int64` key array 复制到 GPU，供 verifier kernel 查询。

hash table 默认 load factor 为 `0.5`，capacity 取不小于 `active_entries / 0.5` 的二次幂。
插入和查询使用相同的 64-bit hash 与 linear probing，默认最多 probe 16 个槽；若构建时
在该 probe 上限内无法容纳全部 key，则 capacity 翻倍后重建。频次只用于决定 key 是否
进入活动表，GPU table 本身不保存 frequency。

因此启动阶段的数据流为：

```text
calibration JSON/JSONL
  -> CPU frequency store
  -> freq_threshold 过滤
  -> CPU open-addressing hash table
  -> GPU active hash table
```

### 2.2 在线 verifier

完成上述 materialization 后，每轮投机解码按以下顺序执行：

1. 投机后端生成一组 draft token；
2. target model 一次验证这些 token；
3. verifier 沿有效路径检查 draft token；
4. 正常验证在某个位置拒绝时，构造 `(draft_token, residual_token)` pair；
5. 查询已经 materialize 到 GPU 的 active hash table；
6. table 命中后继续检查 probability ratio 和可选 entropy gate；
7. 所有门都通过时 force accept draft token，否则保持原始拒绝结果；
8. dynamic 模式把新观察到的 pair 写入 GPU delta buffer。

其中 residual token 是原始 verifier 在拒绝位置准备提交的 target token。CSD key 使用两个 32 位 token id 打包成一个 64 位整数：

```text
key = (draft_token << 32) | residual_token
```

三个后端共用的 runtime、kernel、控制接口及各自适配文件见 1.1 节代码改动总览。

### 2.3 Dynamic rebuild

Dynamic 模式达到 rebuild threshold 后，才把新增 observation 合入上面的转换流程：

```text
GPU delta observations
  -> drain 到 CPU（DP 模式先跨 lane 聚合）
  -> 累加进 CPU frequency store
  -> 更新 active pair 集合
  -> 后台重建 CPU hash table
  -> 复制并替换 GPU active hash table
```

`CSDTableBuilder` 在 CPU 上维护累计频次；某个 pair 的频次首次达到 `freq_threshold` 时，
它才进入 active key 集合。随后 builder 以完整 active key 集合重新生成 hash table，而不是
直接把 delta buffer 当作新表。这样离线 calibration 频次与在线 observation 会持续累加。

hash table 构建在单线程后台 executor 中执行。decode 主线程只负责定期 drain GPU delta、
提交 rebuild，以及在 rebuild 完成后的安全边界把新 table 复制到 GPU 并替换旧 table；它
不会在每次观察到 pair 时同步重建。Plain 模式则只做一次启动 materialization，不执行这一
动态流程。

这里的“验证”包含两个阶段，后文提到的 kernel 切换只发生在第二阶段：

1. **Target verify forward**：target model 对整组 draft token 执行 attention、MLP 并产生 logits；
2. **Accept/rejection verifier**：根据 target logits、draft token 和可选 draft probability 计算接受长度与 bonus token。

CSD 不替换第一阶段的模型计算，也不改变三个后端生成 draft 的 proposer。它只把 table lookup、probability gate、entropy gate 和动态 pair 收集融合到第二阶段的接受 kernel 中。

## 3. 三种 CSD 运行模式

### 3.1 Plain

Plain 只读取离线 calibration table，不在线修改：

```bash
--speculative-csd \
--speculative-csd-table-path <table.json> \
--speculative-csd-freq-threshold 6 \
--speculative-csd-prob-ratio 0.3
```

它没有 rebuild 开销，适合先验证静态 table 的收益。

### 3.2 Dynamic

Dynamic 从离线 table 启动，同时收集线上 pair 并异步 rebuild：

```bash
--speculative-csd-dynamic-update \
--speculative-csd-dynamic-update-ignore-prob-ratio \
--speculative-csd-delta-capacity 16777216 \
--speculative-csd-rebuild-threshold 4096
```

`dynamic-update-ignore-prob-ratio` 只表示“收集 pair 时不使用概率门过滤”。真正 force accept 时仍必须通过 `prob-ratio`。

### 3.3 Dynamic + entropy gate

在 Dynamic 的基础上增加：

```bash
--speculative-csd-force-accept-entropy-threshold <H>
```

只有 target 分布熵不高于 `H` 时才允许 force accept。阈值越低越保守；`-1` 表示关闭 entropy gate。当前实现只在 sampling verifier 中计算完整 target entropy，greedy verifier 不启用该门控，因此评测时必须同时记录采样配置。

## 4. 公共 CSD 参数

### 4.1 ServerArgs 参数注册与启动链路

CSD 参数在 `python/sglang/srt/server_args.py` 中注册；`--speculative-csd` 是
`--speculative-csd-enabled` 的别名。启动时
`python/sglang/srt/arg_groups/speculative_hook.py` 负责检查支持的后端和参数组合，并补全
自动 delta capacity。各 speculative worker 随后用校验后的参数创建自己的
`CSDRuntime`。核心传递关系是：

```text
CLI
  -> ServerArgs 字段
  -> speculative_hook 校验/补默认值
  -> CSDRuntime.from_server_args()
  -> csd_kernel_kwargs()
  -> greedy/sampling verifier CUDA operator
```

只修改 `server_args.py`、`speculative_hook.py` 或 runtime 的 Python 参数传递时，重启服务
即可；若同时修改 operator schema 或 C++/CUDA kernel 参数，则仍需重新构建并安装
`sgl-kernel`。

### 4.2 参数语义

| 参数 | 默认值 | 解释 |
| --- | ---: | --- |
| `--speculative-csd` | 关闭 | 开启 CSD；等价于 `--speculative-csd-enabled` |
| `--speculative-csd-table-path` | 无 | 离线 pair-frequency table。纯动态校准可不提供，从空表启动 |
| `--speculative-csd-freq-threshold` | 6 | pair 累计频次达到该值后才进入活动 hash table |
| `--speculative-csd-prob-ratio` | 0.01 | 要求 `p(draft) / p(residual)` 不低于该值；正式实验显式使用 0.3 |
| `--speculative-csd-dynamic-update` | 关闭 | 在线收集 pair 并启用异步 rebuild |
| `--speculative-csd-dynamic-update-ignore-prob-ratio` | 关闭 | 收集阶段忽略概率比；不影响 force accept 阶段的概率门 |
| `--speculative-csd-delta-capacity` | 自动 | GPU 最多缓存多少条 pair observation；自动值为 `max(2 × rebuild_threshold, 2^20)` |
| `--speculative-csd-rebuild-threshold` | 4096 | buffer 中累计多少 observation 后触发 rebuild；小于等于 0 表示禁用在线 rebuild |
| `--speculative-csd-force-accept-disabled` | 关闭 | 只收集和 rebuild，不改变接受结果；用于 calibration |
| `--speculative-csd-force-accept-entropy-threshold` | -1 | entropy gate 阈值；`-1` 关闭 |
| `--speculative-csd-force-accept-entropy-min-threshold` | -1 | 实验性 entropy 下界；`-1` 关闭。正式方案不使用该参数 |
| `--speculative-csd-delta-save-path` | 无 | 显式导出动态 pair 时使用的可选路径 |

### 4.3 Table、频次与活动表项

JSON calibration table 保存的是 pair 及其累计频次，不是一个只包含“允许接受”键的最终 GPU 表。服务启动或 rebuild 时，只有满足

```text
frequency(draft_token, residual_token) >= freq_threshold
```

的 pair 才进入活动 GPU hash table。因此需要区分：

- `table_store_entries`：CPU frequency store 中所有不同 pair 的数量；
- `table_num_entries`：通过频次阈值、实际可供 verifier 查询的 pair 数量；
- `lookup_hit_ct`：运行时拒绝位置的 pair 命中活动表的次数。

降低 `freq-threshold` 会扩大覆盖率，但也更容易让偶然共现的长尾 pair 进入活动表；提高它会缩小表并提高 pair 的重复证据要求。它只决定 pair 是否可查询，不代表命中后一定 force accept。

`table-path` 在 Plain 模式必需；Dynamic 可以不传 table，从空的 frequency store 启动。离线表与线上 observation 的频次会在 rebuild 时累加，而不是用线上表整体覆盖离线表。

### 4.4 Probability-ratio gate

force accept 的概率门为：

```text
p_target(draft_token) / p_target(residual_token) >= prob_ratio
```

kernel 内部使用等价的 logit 比较，因此不需要额外计算 softmax。以正式实验的 `prob-ratio=0.3` 为例，draft token 在 target 下的概率至少要达到 residual token 的 30%。该值越接近 1 越保守；值越小越容易 force accept，也越可能改变原始采样分布。

这里的 probability ratio 不是“投机成功率”，也不是 calibration table 中 pair 的经验命中率。完整的 force-accept 条件是：原始 verifier 拒绝、table hit、probability-ratio gate 通过，并且可选 entropy gate 也通过。

### 4.5 Dynamic 收集与 rebuild

`--speculative-csd-dynamic-update` 同时启用 GPU delta observation 收集和后台 rebuild。没有该参数时，`rebuild-threshold` 与 `delta-capacity` 不参与运行。

默认情况下，只有通过 probability-ratio gate 的拒绝 pair 才写入 delta buffer。增加 `--speculative-csd-dynamic-update-ignore-prob-ratio` 后，所有观察到的拒绝 pair 都可以进入收集阶段；这个参数**不会**绕过 force-accept 阶段的 probability-ratio gate。它的目的主要是保留更完整的频次分布，方便后续修改门限或导出 calibration table。

`rebuild-threshold` 统计的是 observation 数量，不是不同 pair 的数量。例如阈值 4096 表示缓冲区累计 4096 次观察后可触发 rebuild，同一个 pair 出现多次会重复计数并增加其频次。runtime 每隔若干 decode step 检查一次阈值，因此实际 drain 数量可能略大于该值。

`delta-capacity` 是每个 runtime/rank 上 GPU buffer 可容纳的 observation 数。每条 pair 使用一个 `int64`，因此：

```text
GPU buffer bytes = delta_capacity × 8
```

例如 `16777216` 条约占 128 MiB/rank。capacity 应明显大于 rebuild threshold，为检查间隔、并发请求和跨 DP 聚合留出余量；它不是活动 hash table 的最大表项数，也不会让 CPU frequency store 自动保留同样数量的不同 pair。

rebuild 在单线程后台 executor 中完成，但 drain GPU buffer、把 observation 复制到 CPU，以及完成后把新 hash table 传回 GPU 仍有少量前台工作。过小的 rebuild threshold 会增加调度、拷贝和换表频率；过大则让新 pair 更晚生效。

### 4.6 Calibration-only 与 Entropy Gate

`--speculative-csd-force-accept-disabled` 只关闭“改变接受结果”这一步，不关闭 table lookup、动态收集或 rebuild。校准时使用它可以保证 CSD 不改变生成轨迹；此时应看到 `delta_pair_ct` 增长而 `forced_accept_ct` 始终为 0。

Entropy Gate 只对已经 table hit 且通过 probability-ratio gate 的候选计算 target entropy：

```text
force_accept = previous_gates_passed AND target_entropy <= H
```

所以阈值越低越保守，`-1` 才表示完全关闭。

`delta-save-path` 是动态 observation 显式导出时的可选位置；正式 frequency table 通常通过 `/save_csd_table` 导出。两者都应使用绝对路径，避免多服务或不同工作目录下把结果写到意外位置。

## 5. MTP/EAGLE 接入

### 5.1 原生流程

MTP draft head 按因果顺序逐步提出 token，SGLang 复用 EAGLE worker 和 tree verifier。Qwen3.5 的 MTP 权重已包含在模型 checkpoint 中，因此通常不需要单独指定 draft model path。

当前正式测试采用 `3-1-4`：

```bash
--speculative-algorithm EAGLE \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

三个数字分别表示：

- `num-steps=3`：draft head 展开的深度；
- `eagle-topk=1`：每个 draft step 只保留一个候选分支；
- `num-draft-tokens=4`：target verifier 每轮最多处理的 draft token 数。

### 5.2 CSD 插入点

`EAGLEWorkerV2` 创建公共 `CSDRuntime`，并在 `eagle_utils.py` 调用 verifier 时传入 table、delta buffer 和门控参数。

- Greedy：bare 与 CSD 都调用 `sgl_kernel.verify_tree_greedy`；CSD 是在同一个 CUDA tree verifier 中启用额外参数，不是替换 target forward；
- Sampling、未启用 rejection sampling：bare 与 CSD 都调用 `sgl_kernel.tree_speculative_sampling_target_only`，CSD 在该 CUDA verifier 内增加 lookup 和门控；
- Sampling、显式启用 rejection sampling：bare 使用 `chain_speculative_sampling_triton` 快路径，CSD 切换到支持同一 `p/q` 接受规则的 `tree_speculative_sampling_target_only` CUDA verifier；
- 每轮 decode 前检查上一轮异步 rebuild 是否完成，并判断是否需要启动新 rebuild。

可以概括为：

```text
target verify forward（不变）
  -> bare accept kernel
  -> 开启 CSD 后：相同 greedy CUDA verifier，或带 CSD 的 CUDA sampling verifier
```

因此“bare MTP 保留 chain Triton 快路径”只适用于显式开启 rejection sampling 的 sampling 配置，不能泛化到所有 MTP 运行。当前正式精度脚本没有开启该选项，走的是 target-only sampling verifier。

MTP 的 proposal 是因果生成的，因此相同 token pair 通常比 diffusion block proposal 更集中，现有实验中也是三种后端里 calibration table 复用最稳定的一种。

### 5.3 启动示例

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path /data/model/Qwen3.5-35B-A3B \
  --tp 4 --trust-remote-code \
  --max-running-requests 48 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-csd \
  --speculative-csd-table-path <mtp_table.json> \
  --speculative-csd-freq-threshold 6 \
  --speculative-csd-prob-ratio 0.3 \
  --host 127.0.0.1 --port 32120
```

本轮两组 GPU 的完整测试入口是：

```text
runs/mtp_csd/qwen35b_mtp314_final/run_all.sh
```

## 6. DFlash 接入

### 6.1 原生流程

DFlash 使用独立 diffusion draft checkpoint，一次并行提出固定长度 block，再由 target model 验证。它不使用 EAGLE 的多步、多分支树，因此代码会强制：

```text
speculative_num_steps = 1
speculative_eagle_topk = 1
```

### 6.2 DFlash 独有参数

| 参数 | 解释 |
| --- | --- |
| `--speculative-draft-model-path` | DFlash draft checkpoint，必须提供 |
| `--speculative-dflash-block-size` | proposal block 和 verify window 长度；也是 DFlash 下 `num-draft-tokens` 的别名 |
| `--speculative-num-draft-tokens` | 可替代 block-size，但两者同时提供时必须相等 |
| `--speculative-draft-window-size` | 可选 draft 工作窗口，不能小于 block size |

如果未显式设置 block size，代码先尝试从 draft checkpoint 配置读取；读取失败时回退到 16。DFlash 当前不支持 DP Attention，也要求 `pp-size=1`。

### 6.3 CSD 插入点

`DFlashWorkerV2` 创建 `CSDRuntime`，greedy 和 sampling 接受路径分别在 `dflash_utils.py` 中接入公共 CUDA verifier。

- Greedy bare：target forward 之后调用 DFlash 专用 `_compute_dflash_accept_bonus_triton_unchecked`，一次计算连续匹配长度、bonus、commit length 和输出 token；
- Greedy + CSD：退出上述专用 Triton accept/bonus 分支，改用公共 `sgl_kernel.verify_tree_greedy` CUDA verifier 执行线性链验证和 CSD lookup，随后再组装 commit/output；
- Sampling bare：调用公共 `sgl_kernel.tree_speculative_sampling_target_only` 计算接受长度和 residual/bonus token；
- Sampling + CSD：仍调用 `tree_speculative_sampling_target_only`，但传入 CSD table、delta buffer、probability gate 和 entropy gate；
- table metadata 会检查 backend 和 block size，避免加载 MTP、DSpark 或不同 block size 的表。

DFlash 的 diffusion proposer、target verify forward 和 block attention mask 均保持不变。发生变化的是 logits 产生后的 accept/bonus 阶段；所以 CSD greedy 路径会失去 bare 专用 Triton 融合带来的部分优势，而 sampling 路径主要是在已有 CUDA verifier 内增加 CSD 工作。

DFlash 每个 block 后部位置的 proposal 条件与 target 自回归路径并不完全一致，因此必须用 DFlash 自己采集的 table，不能直接复用 MTP table。

### 6.4 启动参数

```bash
--speculative-algorithm DFLASH \
--speculative-draft-model-path /data/model/Qwen3.5-35B-A3B-DFlash \
--speculative-dflash-block-size 16
```

## 7. DSpark 接入

### 7.1 原生流程

DSpark 也使用并行 diffusion proposal，但它在 DFlash block 基础上增加 confidence 估计和 compact ragged verify：每个请求可以根据置信度和系统预算选择不同的有效 verify 长度，避免所有请求始终验证完整 block。

DSpark 的一轮 decode 是：

```text
diffusion proposal
  -> confidence/STS
  -> SPS budget 与 compact layout
  -> target ragged verify
  -> accept/commit
```

### 7.2 DSpark 独有参数

| 参数 | 解释 |
| --- | --- |
| `--speculative-dspark-block-size <gamma>` | 实际提出的 draft token 数 gamma |
| `--speculative-num-draft-tokens` | DSpark 中表示 verify window，必须等于 `gamma + 1` |
| `--speculative-dspark-sps-table-path` | 离线 profile 的系统开销表，compact scheduler 用它计算 verify token 预算 |
| `--speculative-dspark-confidence-sts-path` | confidence head 的逐位置 STS 校准表；只校准 survival probability |
| `--speculative-dspark-align-verify-tokens-to-graph-tier` | 用真实 verify token 填充本来就要支付的 CUDA Graph padding tier |
| `SGLANG_RAGGED_VERIFY_MODE=compact` | 选择 compact ragged-verify 模式 |

需要特别区分三种表：

- SPS table：描述不同 batch/verify token 数的系统成本；
- STS table：校准 confidence；
- CSD table：统计 `(draft_token, residual_token)` pair 频次。

它们用途完全不同，不能互相替代。省略 SPS table 时，预算会退化为 verify-all，本身不会产生 compact 吞吐收益；省略 STS table 不影响 lossless 正确性，但可能使预算分配不准。

当前模型把 DSpark draft 权重打包在 target checkpoint 中，代码可自动令 `draft-model-path=model-path`。DSpark 要求 `pp-size=1`。使用 DP Attention 时还要求：

```bash
--tp 4 --dp-size 4 \
--enable-dp-attention --enable-dp-lm-head \
--moe-a2a-backend none
```

这里不是同时复制四份完整模型：模型权重仍由 TP 组织，DP Attention 把 attention 请求分到不同 lane，MoE/其余权重计算仍按当前模型并行布局执行。

### 7.3 CSD 插入点

`DSparkWorkerV2` 在 compact/non-compact target verify 完成后，把 `CSDRuntime` 传给 `accept_draft_tokens()`：

- CSD 只作用于 compact 剪枝后仍被 target 验证的 token；已经被 cap-len 剪掉的 token 不会进入 CSD；
- Greedy bare：普通路径使用 DSpark 的 `accept_greedy_triton`；满足 compact、CUDA Graph 和无额外 logits adjustment 等条件时，还可把 accept/finalize/部分 commit 折叠进 `DsparkVerifyEpilogue`；
- Sampling bare：普通路径使用 `chain_speculative_sampling_triton`；
- Greedy + CSD：切换到公共 `sgl_kernel.verify_tree_greedy` CUDA verifier；
- Sampling + CSD：切换到公共 `sgl_kernel.tree_speculative_sampling_target_only` CUDA verifier，并保留 rejection-sampling 的 `p/q` 规则；
- CSD 开启时 `fold_eligible=False`，folded accept/commit epilogue 被禁用，因为接受阶段还需要 table lookup、entropy 计算和状态更新；target 的 compact ragged forward 本身仍可照常运行；
- DP Attention 下，各 lane 的 delta pair 会跨 DP 聚合，再让各 rank rebuild 出一致 table；
- DP 聚合只保留 attention-TP leader 的一份 observation，避免同一 lane 的 TP 副本重复计数。

所以 DSpark 的路径变化可以写成：

```text
bare compact: target CUDA Graph -> 可选 folded Triton accept/finalize/commit
CSD compact : target compact verify -> 公共 CSD CUDA verifier -> finalize/commit
```

这不是把 DSpark 的 compact verify 改回 verify-all，也不是替换 target model kernel；只是接受判定不能继续折叠在原 CUDA Graph epilogue 中。混合 greedy/sampling 的异构 batch 为避免重复记录未选中的分支，目前保持 baseline 接受语义；正式实验使用同质 sampling batch，可完整启用 CSD。

DSpark 一次并行生成整块 proposal，后部 token 与 target 真实因果路径的对应更弱；compact 又进一步减少实际进入 verifier 的 token。因此即使 raw table hit 不低，能同时通过 target probability gate 并转化为 force accept 的比例也可能低于 MTP。

### 7.4 启动示例

```bash
GPU_SET=4,5,6,7 PORT=30000 \
  bash scripts/run_dspark_csd.sh plain-server
```

脚本会加载：

```bash
--speculative-algorithm DSPARK \
--speculative-dspark-sps-table-path <sps_table.json>
```

以及 DSpark 的 TP/DP Attention、compact 模式和 CSD 参数。

## 8. Calibration 与 table 导出

三个后端的 calibration 过程相同，但必须分别采集 table：

1. 使用与正式实验相同的模型、投机后端和 draft 形状启动服务；
2. 开启 dynamic update；
3. 开启 `force-accept-disabled`，保证校准不改变生成结果；
4. 运行 calibration prompts；
5. 调用 `/save_csd_table`；
6. 正式评测时关闭 `force-accept-disabled` 并加载导出的 table。

本项目默认使用 RedPajama 六域 calibration 集。六个域分别为：

```text
arxiv
c4
common_crawl
github
stackexchange
wikipedia
```

每个域取 1000 条 prompt，共 6000 条。所有正式table都使用`temperature=1.0`和
`top_p=1.0`，但历史采集长度并不完全相同：Qwen3.5-35B和397B的现存MTP table使用
`max_new_tokens=512`、并发8；DSpark和DeepSeek-V4 MTP的正式校准使用
`max_new_tokens=1024`。DFlash脚本使用48个并发请求，DSpark通用脚本使用8个并发请求；
并发数只影响校准耗时，不改变语料组成。校准文件虽然包含历史 completion，但
`scripts/calibrate_dspark_csd.py` 只读取其中的 `domain` 和 `prompt`，由当前服务重新生成
completion 并采集 pair。

当前默认 prompt 文件为：

```text
/root/sglang-csd-archive-20260722/runs_legacy/redpajama/redpajama_csd_calibration_answer.jsonl
```

MTP、DFlash 和 DSpark 可以使用同一份 6000 条 prompt 作为输入，但不能共用导出的
table。table 依赖 proposer 的条件分布和 draft 形状，因此至少应分别记录：target model、
draft model/后端、MTP steps/topk/draft tokens、DFlash block size、DSpark SPS/compact 配置，
以及 temperature、top-p 和最大生成长度。任一关键项变化后都应重新校准，或明确将旧表
作为跨配置迁移实验，而不能作为同配置结果报告。

校准服务的核心参数为：

```bash
--speculative-csd \
--speculative-csd-dynamic-update \
--speculative-csd-dynamic-update-ignore-prob-ratio \
--speculative-csd-force-accept-disabled \
--speculative-csd-freq-threshold 6 \
--speculative-csd-prob-ratio 0.01 \
--speculative-csd-delta-capacity 16777216 \
--speculative-csd-rebuild-threshold 4096
```

上面的`freq-threshold=6`是正式评测加载table时使用的活动表准入阈值。两个Qwen table的
metadata中记录的`freq_threshold=3`表示：当时运行calibration服务时，服务用阈值3构建
运行期GPU活动表。该值不是采集过滤条件，也不是JSON导出过滤条件；导出的JSON仍包含
频次1、2以及更高频次的完整CPU pair-frequency store。正式评测服务读取同一JSON后，
根据当前命令行参数重新筛选`frequency >= 6`的pair并构建GPU活动表。

导出示例：

```bash
curl -fsS -X POST http://127.0.0.1:<port>/save_csd_table \
  -H 'Content-Type: application/json' \
  -d '{"path":"/absolute/path/csd_table.json","metadata":{"backend":"mtp"}}'
```

DSpark 的通用 calibration 流程保存在 `scripts/run_dspark_csd.sh`。DP=4 的 calibration
会先导出四个 rank shard，再由
`scripts/merge_csd_tables.py` 按 pair 对频次求和。

### 8.1 Calibration 脚本与 table 索引

下面只列出本次中期报告实际引用的 calibration 入口和 table。路径均相对于仓库根目录
`/root/sglang-dspark-csd`。

| Target model / 后端 | Draft 配置 | Calibration 脚本 | 正式或现存 table | 状态与用途 |
| --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B / MTP | EAGLE 3-1-4 | `runs/mtp_csd/qwen35b_mtp314_final/calibration/run_qwen35_mtp314_calibration.sh` | `runs/mtp_csd/qwen35b_mtp314_final/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-35B-A3B_mtp_EAGLE_steps3_topk1_draft3_temp1.0_ratio0.01.json` | 已用于当前 Qwen 正式评测；历史文件名和 CLI metadata 中的 `draft3` 表示 3 个 draft token，计入 verifier root 后对应报告中的 4-token 有效宽度，因此统一记作 3-1-4 |
| Qwen3.5-397B-A17B-FP8 / MTP | EAGLE 3-1-4 | 复用上一行脚本并显式覆盖`MODEL`、`TABLE`与输出目录 | `runs/mtp_csd/qwen397b_mtp314_final/calibration/csd_table_redpajama_logits_ungated_6domains_n1000_Qwen3.5-397B-A17B-FP8_mtp_EAGLE_steps3_topk1_draft3_temp1.0_ratio0.01.json` | 6000 prompts、512 max-new-tokens、并发8；正式加载阈值6 |
| DeepSeek-V4-Flash-DSpark / DSpark | compact DSpark，SPS table，DP=4 | `scripts/run_dspark_csd.sh` | `runs/dspark_csd/formal_redpajama_20260724/tables/dspark_csd_merged.json` | 已完成；四个 DP shard 合并后的正式 DSpark table |
| DeepSeek-V4-Flash / MTP | EAGLE 3-1-4，DeepSeek-V4-Flash-MTP-Draft，DP=4 | `runs/dspark_csd/v4_mtp314_calibration_dp4_20260802/run_full_calibration.sh` | `runs/dspark_csd/v4_mtp314_calibration_dp4_20260802/tables/v4_mtp314_redpajama_merged.json` | 已完成；6000 prompts，四 shard 频次守恒校验通过 |

Qwen MTP 和 DSpark 的 calibration 客户端共用：

```text
scripts/calibrate_dspark_csd.py
```

尽管文件名保留了最初的 DSpark 命名，该客户端只负责读取固定 prompt、向指定 SGLang
服务发送生成请求并汇总 token/verification 指标，不包含 DSpark 专用 proposer 逻辑，因此也
可用于 MTP。各后端的 proposer、draft 形状和 CSD 收集行为由服务端启动脚本决定。

DSpark 的完整校准顺序为：

```bash
bash scripts/run_dspark_csd.sh formal-calibration-server
bash scripts/run_dspark_csd.sh formal-calibration-run
bash scripts/run_dspark_csd.sh export-table
```

DeepSeek-V4 MTP 3-1-4 使用单一编排脚本完成服务启动、6000 条请求、四 shard 导出、合并
和频次守恒校验：

```bash
bash runs/dspark_csd/v4_mtp314_calibration_dp4_20260802/run_full_calibration.sh
```

## 9. 运行指标与报告口径

运行时通过下面的接口检查：

```bash
curl -fsS http://127.0.0.1:<port>/server_info | python -m json.tool
```

建议保留以下运行指标：

| 指标 | 含义 |
| --- | --- |
| `csd_lookup_hit_ct` | 被验证位置的 pair 命中活动 table 的次数 |
| `csd_forced_accept_ct` | 最终通过所有门并 force accept 的次数 |
| `csd_delta_pair_ct` | 写入 delta buffer 的 observation 数 |
| `csd_table_num_entries` | 当前活动 GPU hash table 表项数 |
| `csd_table_store_entries` | CPU frequency store 中的不同 pair 数 |
| `csd_rebuild_started_ct` | 已启动的异步 rebuild 次数 |
| `csd_rebuild_applied_ct` | 已替换到 GPU 的新 table 次数 |
| `csd_rebuild_wall_sec` | 后台 rebuild 累计耗时 |
| `csd_dp_aggregate_pair_ct` | DP Attention 下跨 lane 聚合的 pair 数 |

### 9.1 正式结果脚本索引

| 模型 / 后端 | 正式任务 | 正式结果脚本 | 主要输出 |
| --- | --- | --- | --- |
| Qwen3.5-35B-A3B / MTP | AIME 2025、Math500、LiveCodeBench v6、GSM8K | `runs/mtp_csd/qwen35b_mtp314_final/run_all.sh` | `runs/mtp_csd/qwen35b_mtp314_final/results/{aime25_avg16,math500_avg4,lcb_avg4,gsm8k_avg4}/` |
| Qwen3.5-35B-A3B / MTP | APPS、TACO，各题 n=4 | `runs/mtp_csd/qwen35b_mtp314_final/run_apps_taco_n4.sh` | `runs/mtp_csd/qwen35b_mtp314_final/results/{apps,taco}/` |
| Qwen3.5-397B-A17B-FP8 / MTP | AIME 2025、Math500、LiveCodeBench v6、GSM8K | `runs/mtp_csd/qwen397b_mtp314_final/run_all.sh` | `runs/mtp_csd/qwen397b_mtp314_final/results/` |
| DeepSeek-V4-Flash-DSpark / DSpark | LCB avg@4、AIME avg@16、Math500 avg@4、GSM8K avg@4，static四方法 | `runs/dspark_csd/final_results/run_all.sh` | `runs/dspark_csd/final_results/<method>/<task>/` |

### 9.2 Qwen3.5-35B MTP

配置：`steps=3`、`topk=1`、`draft_tokens=4`、`max_running_requests=48`、
`temperature=1.0`、`top_p=0.95`、`top_k=20`、`presence_penalty=1.5`、thinking、
`max_new_tokens=81920`、`model_length=96000`。

| 任务 | 报告指标 |
| --- | --- |
| AIME 2025 | avg@16、pass@16 |
| Math500 | avg@4、pass@4 |
| LiveCodeBench v6 | avg@4、pass@4 |
| GSM8K | avg@4、pass@4 |
| APPS、TACO | avg@4、pass@4 |

```bash
bash runs/mtp_csd/qwen35b_mtp314_final/run_all.sh all
bash runs/mtp_csd/qwen35b_mtp314_final/run_apps_taco_n4.sh all
```

```text
runs/mtp_csd/qwen35b_mtp314_final/results/
```

主任务目录包含`artifacts/`、`logs/`和`results/classic_tree_shape_sweep.jsonl`；
APPS/TACO目录包含`answers/`、`accuracy/`和`logs/`。Math500 entropy使用p30阈值
`1.3415851593017578`，其余entropy结果使用p20阈值`1.5638477802276611`。

### 9.3 Qwen3.5-397B MTP

```bash
bash runs/mtp_csd/qwen397b_mtp314_final/run_all.sh all
```

| 任务 | 报告指标 | CSD probability ratio |
| --- | --- | ---: |
| AIME 2025 | avg@4、pass@4 | 0.3 |
| Math500 | avg@4、pass@4 | 0.4 |
| LiveCodeBench v6 | avg@4、pass@4 | 0.3 |
| GSM8K | avg@4、pass@4 | 0.3 |

输出目录：

```text
runs/mtp_csd/qwen397b_mtp314_final/results/{aime25_avg4,math500_avg4,lcb_avg4,gsm8k_avg4}/
```

### 9.4 DSpark

配置：DeepSeek-V4-Flash-DSpark、TP=8、DP=8、static verify、bare/plain/dynamic/entropy
四种方法、`prob-ratio=0.3`、entropy阈值`1.5638477802276611`。

| 任务 | 报告指标 |
| --- | --- |
| AIME 2025 | avg@16、pass@16 |
| Math500 | avg@4、pass@4 |
| LiveCodeBench v6 | avg@4、pass@4 |
| GSM8K | avg@4、pass@4 |

```bash
bash runs/dspark_csd/final_results/run_all.sh
```

```text
runs/dspark_csd/final_results/<method>/<task>/
```

每个任务目录包含`eval/result.json`、`eval/tracker/`和`metrics/task_metrics.json`。
