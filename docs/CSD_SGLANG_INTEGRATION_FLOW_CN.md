# CSD 在 SGLang 中的接入位置与执行流程

本文只描述当前仓库代码实际实现的接入关系，目的是为技术报告中的系统流程图提供依据。

## 1. 结论：CSD 接在 verifier 的“正常拒绝”分支

CSD 不修改 MTP、DFlash 或 DSpark 的 draft 生成，也不修改 target model 的前向计算。三个后端仍先生成各自形状的候选 token，再由 target model 一次计算验证所需的 logits/probabilities。

CSD 的插入点位于 speculative verifier 内部：原生规则判定当前 draft token 未通过之后、verifier 输出 residual/target token 之前。只有正常拒绝才触发 CSD；正常接受的 token 不查询 CSD table。

```text
draft backend 生成候选
        -> target forward 得到 logits
        -> 原生 verifier 判断当前 draft token
        -> 正常接受：保持原路径
        -> 正常拒绝：进入 CSD rescue
        -> force accept draft，或保持原拒绝并输出 residual/target token
```

因此，CSD 是 verifier 的拒绝后处理扩展，而不是第四种 speculative backend。

## 2. 三个后端的代码接入

三个 worker 分别创建一份统一的 `CSDRuntime`：

| 后端 | runtime 创建位置 | 验证接入位置 |
| --- | --- | --- |
| MTP/EAGLE | `python/sglang/srt/speculative/eagle_worker_v2.py` | `python/sglang/srt/speculative/eagle_utils.py` |
| DFlash | `python/sglang/srt/speculative/dflash_worker_v2.py` | `python/sglang/srt/speculative/dflash_utils.py` |
| DSpark | `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` | `.../kernels/accept_greedy.py` 与 `accept_sampling.py` |

统一状态和参数适配位于：

```text
python/sglang/srt/speculative/csd_runtime.py
  CSDRuntime.from_server_args()
  csd_kernel_kwargs()
```

`csd_kernel_kwargs()` 把 GPU hash table、delta buffer、计数器、probability ratio 和 entropy threshold 传入 verifier operator。三个后端共享相同的 table、gate、指标和动态更新语义，但候选布局和原生 verifier 路径并不完全相同。

## 3. Greedy 与 sampling 的实际 kernel 路径

### 3.1 Greedy

MTP/EAGLE 使用 `verify_tree_greedy`；DFlash 和 DSpark 在开启 CSD 时，也把线性候选适配成 chain-shaped tree buffer 后调用该 operator。其 CUDA 实现在：

```text
sgl-kernel/csrc/speculative/eagle_utils.cu
```

正常接受条件为 `draft_token == target_top1_token`。拒绝时构造：

```text
(draft_token, target_top1_token)
```

然后检查 table hit 与 logit margin。当前 greedy operator **没有接入 entropy gate**。

### 3.2 Sampling

开启 CSD 后，MTP/EAGLE、DFlash 和 DSpark 的 sampling 验证最终都调用：

```text
tree_speculative_sampling_target_only
  -> sgl-kernel/csrc/speculative/speculative_sampling.cu/.cuh
```

DSpark 在不开 CSD时保留 `chain_speculative_sampling_triton` 快路径；开启 CSD 后切换到上述 CUDA verifier。MTP/EAGLE 在 rejection-sampling 快路径且不开 CSD时也可走 Triton；CSD 开启时切换到带 CSD 参数的 CUDA verifier。DFlash 使用相同 CUDA verifier 处理其线性候选。

sampling 正常拒绝后，kernel 先按原 verifier 规则采样 residual token，并构造：

```text
(draft_token, residual_token)
```

随后在同一个 CUDA verifier 内完成 table lookup、probability-ratio gate、entropy gate 和 force accept，不需要把拒绝位置或 logits 搬回 Python。

## 4. 一次 CSD rescue 的准确逻辑

以 sampling 路径为主，执行顺序为：

1. 原生 verifier 计算 `normal_accept`；只有结果为 false 才进入 CSD。
2. 按原拒绝逻辑采样 residual token，并得到 residual 对应的最大 target logit。
3. 打包 `(draft_token, residual_token)` 为 64-bit key。
4. 查询 GPU active hash table；命中时累计 `lookup_hit_ct`。
5. 检查 logit margin：`draft_logit >= residual_logit + log(prob_ratio)`。
6. 若配置 entropy gate，计算该位置 target distribution 的 entropy，并检查上下界。当前实验采用上界保护：entropy 高于阈值时不允许 force accept。
7. table、ratio、entropy 和 `force_accept_disabled` 状态全部允许时，累计 `forced_accept_ct` 并接受 draft token。
8. 否则保持原始拒绝结果，沿用 residual token。

注意：table frequency 只在启动/重建阶段决定哪些 pair 进入 active hash table；GPU verifier 查询的 hash table 只保存 key，不保存 frequency。

## 5. Dynamic update 是旁路反馈，不属于前向主链

在正常拒绝位置，kernel 可把新观察到的 pair 追加到 GPU delta buffer。默认情况下 pair 还要通过 ratio gate；开启 `dynamic_update_ignore_prob_ratio` 时则忽略该限制。该采集不要求 pair 已存在于 active table。

每个 worker 在后续 `forward_batch_generation()` 开始时驱动 rebuild 状态机：

1. 定期检查 delta 数量；
2. DP attention 模式下，由 attention-TP leader 提交本 lane 数据，并通过 model TP group 聚合各 DP lane；
3. 将 delta pairs 交给后台 CPU 线程累计 frequency；
4. 按 `freq_threshold` 重新筛选 active keys，并构建 CPU hash table；
5. 在后续 step 将新 key array materialize 到 GPU，原子式替换 runtime 当前 table。

因此 dynamic update 应在系统图中画成从 verifier rejection 分支返回 CSD table 的虚线反馈环，而不应画成每个 token 都同步阻塞的主流程。

## 6. PPT 流程图

可编辑的 Graphviz 源文件：

```text
docs/figures/csd_sglang_integration_flow.dot
```

渲染结果：

```text
docs/figures/csd_sglang_integration_flow.svg
docs/figures/csd_sglang_integration_flow.png
```

这张图刻意区分了三层：后端 proposal、target verifier 主链、dynamic update 旁路。PPT 中建议把它放在现有 CSD 决策流程页之后，标题使用“CSD 在 SGLang 多投机后端中的统一接入”。
