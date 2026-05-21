# CSD 表质量与 LCB 精度分析

本文整理 `/home/zhouxuwen/sglang/benchmark` 下已有的 CSD table 和评测结果。

## 总结

当前结果说明：logits-gated calibration 确实能生成更干净的 CSD table，但更干净的 table 仍然不足以恢复 LCB / CoT 类任务的精度。

一句话结论：

> table 确实比之前更干净了，但是 `干净 table + logits gate` 仍然不能保证 LCB/code-generation 任务的正确性。现在主要问题看起来更像是 force-accept 策略本身，尤其是在高熵、长 CoT、代码生成位置上，而不只是 table 太脏。

## 1. Table 规模说明清洗确实有效

从 table metadata 和 entries 数量看，更严格的 logits-gated calibration 明显减少了 table 大小。

| Table | Calibration ratio | Spec shape | entries | total freq | freq>=3 | freq>=6 |
|---|---:|---|---:|---:|---:|---:|
| 原始 RedPajama table | 0.01 | steps5/topk3/draft15 | 1,380,298 | 2,725,858 | 112,179 | 40,520 |
| Logits-gated RedPajama ratio0.01 | 0.01 | steps5/topk3/draft15 | 1,220,125 | 2,461,343 | 100,040 | 36,246 |
| Logits-gated RedPajama ratio0.3 | 0.3 | steps5/topk3/draft15 | 692,510 | 1,456,381 | 57,161 | 21,178 |
| Logits-gated RedPajama ratio1 | 1.0 | steps5/topk3/draft15 | 357,450 | 724,948 | 27,151 | 10,158 |
| Logits-gated RedPajama ratio0.3 | 0.3 | steps3/topk1/draft3 | 431,186 | 887,466 | 33,523 | 12,513 |
| Logits-gated RedPajama ratio1 | 1.0 | steps3/topk1/draft3 | 278,983 | 576,868 | 21,464 | 8,162 |
| LCB v1 calibration table | 0.3 | steps5/topk3/draft15 | 15,483~31,538 | 28,134~60,706 | 1,628~3,465 | 606~1,316 |

观察：

- calibration ratio 从 `0.01 -> 0.3 -> 1.0` 增大时，table entries 显著下降。
- 使用更小的 speculative shape，即 `steps3/topk1/draft3`，也会进一步减少 table 大小。
- 因此从 table 统计角度看，table cleaning 机制是生效的。

## 2. LCB 结果：更干净的 table 仍然无法恢复精度

### 2.1 Baseline / vanilla / CSD 对比

主要来源：

```text
/home/zhouxuwen/sglang/benchmark/csd/runs/lighteval/results/result.jsonl
```

| Mode | Task | Spec shape | Table | Eval ratio | Dynamic update | Accuracy |
|---|---|---|---|---:|---|---:|
| baseline | lcb:codegeneration_v6 | none | none | - | no | **0.788571** |
| vanilla EAGLE | lcb:codegeneration_v6 | steps5/topk3/draft15 | none | - | no | 0.742857 |
| vanilla EAGLE | lcb:codegeneration_v6 | steps5/topk1/draft5 | none | - | no | 0.720000 |
| CSD | lcb:codegeneration_v6 | steps5/topk3/draft15 | old RedPajama table | 0.3 | no | 0.720000 |
| CSD | lcb:codegeneration_v6 | steps5/topk3/draft15 | old RedPajama table | 0.3 | yes | 0.645714 |
| CSD | lcb:codegeneration_v6 | steps5/topk1/draft5 | old RedPajama table | 0.3 | no | 0.714286 |
| CSD | lcb:codegeneration_v6 | steps5/topk1/draft5 | old RedPajama table | 0.3 | yes | 0.645714 |

这一组体现出基本的失败模式：

```text
baseline > vanilla speculative > static CSD > dynamic CSD
```

baseline 达到 `0.788571`，vanilla EAGLE 降到约 `0.742857`，CSD 约 `0.72` 或更低。dynamic CSD 更差，约 `0.645714`。

### 2.2 Logits-gated table 对比

| Mode | Task | Table | Table calibration ratio | Eval ratio | Spec shape | Dynamic update | Accuracy |
|---|---|---|---:|---:|---|---|---:|
| CSD | LCB v6 | logits-gated ratio0.01 | 0.01 | 0.3 | steps5/topk3/draft15 | no | 0.714286 |
| CSD | LCB v6 | logits-gated ratio0.3 | 0.3 | 0.3 | steps5/topk3/draft15 | no | 0.731429 |
| CSD | LCB v6 | logits-gated ratio1 | 1.0 | 0.3 | steps5/topk3/draft15 | no | 0.702857 |
| CSD | LCB v6 | logits-gated ratio1 | 1.0 | 0.3 | steps5/topk3/draft15 | yes | 0.651429 |
| CSD | LCB v6 | logits-gated ratio1 | 1.0 | 1.0 | steps3/topk1/draft3 | no | 0.725714 |
| CSD | LCB v6 | logits-gated ratio1 | 1.0 | 1.0 | steps3/topk1/draft3 | yes | 0.731429 |
| CSD | LCB v6 | logits-gated ratio0.3 | 0.3 | 0.3 | steps3/topk1/draft3 | no | 0.737143 |
| CSD | LCB v6 | logits-gated ratio0.3 | 0.3 | 0.3 | steps3/topk1/draft3 | yes | 0.628571 |

关键点：

- logits-gated ratio0.3 table 在 LCB 上大约是 `0.731429~0.737143`。
- logits-gated ratio1 table 并不稳定更好，大约是 `0.702857~0.731429`。
- 这些结果仍然低于 baseline `0.788571`。
- 它们也大多低于或接近 vanilla EAGLE `0.742857`。

结论：

> 更干净的 table 没有解决 LCB 精度问题。

## 3. 另一组 LCB 设置也显示同样的顺序

还有另一组 LCB 结果中，baseline 本身更低，可能是生成参数、温度或任务配置不同导致的。但排序仍然一致。

| Mode | Task | Accuracy |
|---|---|---:|
| baseline | LCB v6 | 0.594286 |
| vanilla EAGLE steps5/topk3/draft15 | LCB v6 | 0.577143 |
| vanilla EAGLE steps5/topk1/draft5 | LCB v6 | 0.560000 |
| CSD static steps5/topk3/draft15 | LCB v6 | 0.462857 |
| CSD dynamic steps5/topk3/draft15 | LCB v6 | 0.445714 |
| CSD static steps5/topk1/draft5 | LCB v6 | 0.462857 |
| CSD dynamic steps5/topk1/draft5 | LCB v6 | 0.422857 |

仍然是：

```text
baseline > vanilla speculative > CSD static > CSD dynamic
```

## 4. GSM8K / AIME / Math500 结果说明机制不是全局坏掉

这个问题不像简单的全局 CUDA/kernel bug，因为 CSD 在 GSM8K、AIME25、Math500 上并不总是坏的；不同任务的敏感性明显不同。

### 4.1 LightEval GSM8K

| Source | Mode | Accuracy |
|---|---|---:|
| LightEval baseline | baseline | 0.918878 |
| LightEval vanilla | vanilla | 0.921911~0.922669 |
| LightEval CSD | CSD | 0.891585~0.930250 |

LightEval GSM8K 上 CSD 方差较大。它有时更差，但也能达到 `0.930250`。

### 4.2 lm_eval GSM8K

| Source | Mode | Accuracy range |
|---|---|---:|
| lm_eval baseline | baseline | 0.9340~0.9431 |
| lm_eval vanilla | vanilla | 0.9333~0.9431 |
| lm_eval CSD | CSD | 0.9386~0.9522 |

在 lm_eval GSM8K 上，CSD 甚至能略微提升：

```text
baseline avg ≈ 0.9368
vanilla avg ≈ 0.9400
CSD avg ≈ 0.9409
```

### 4.3 LightEval AIME25

主要来源：

```text
/home/zhouxuwen/sglang/benchmark/csd/runs/lighteval/results/result.jsonl
/home/zhouxuwen/sglang/benchmark/csd/runs/lighteval_batchsize_sweep/*/results/result.jsonl
```

| Source | Mode | Setting | Accuracy |
|---|---|---|---:|
| LightEval main result | baseline | reference run | 0.933333 |
| LightEval main result | vanilla | steps5/topk3/draft15 or steps5/topk1/draft5 | 0.900000 |
| LightEval main result | CSD | multiple static/dynamic CSD runs | 0.800000~0.933333 |
| LightEval batch sweep | vanilla | repeated batch/spec-shape sweep | 0.866667~0.933333, avg ≈ 0.91 |
| LightEval batch sweep | CSD | repeated batch/spec-shape sweep | 0.800000~0.933333, avg ≈ 0.86~0.87 |

AIME25 的样本数很小，`30` 题中一道题就是 `3.33%`，所以 `0.90` 和 `0.933333` 只差一道题，不能过度解释单次 run 的差异。

但从多组结果看，AIME25 上的趋势大致是：

```text
baseline / vanilla 较稳，CSD 有时接近 baseline，但均值通常略低，dynamic 不稳定。
```

这说明 CSD 在数学推理类任务上也可能有精度风险，但没有 LCB 那么系统性、剧烈。

### 4.4 LightEval Math500

主要来源：

```text
/home/zhouxuwen/sglang/benchmark/csd/runs/lighteval/results/result.jsonl
/home/zhouxuwen/sglang/benchmark/csd/runs/lighteval/results/result_low_ratio.jsonl
```

| Source | Mode | Setting | Accuracy |
|---|---|---|---:|
| LightEval main result | baseline | reference run | 0.854 |
| LightEval main result | vanilla | steps5/topk3/draft15 | 0.852 |
| LightEval main result | vanilla | steps5/topk1/draft5 | 0.824 |
| LightEval main result | CSD | multiple static/dynamic CSD runs | 0.812~0.854, avg ≈ 0.838 |
| LightEval low-ratio result | CSD static | ratio0.01 | 0.858 |
| LightEval low-ratio result | CSD dynamic | ratio0.01 | 0.818 |

Math500 上 CSD 不像 LCB 那样大幅崩掉。静态 CSD 可以接近 baseline，个别 ratio0.01 静态结果甚至到 `0.858`，略高于 baseline `0.854`。但 dynamic CSD 仍然偏危险，例如 ratio0.01 dynamic 为 `0.818`。

Math500 的结论更接近：

```text
静态 CSD 基本可用但有波动；dynamic update 更容易掉点。
```

### 4.5 跨任务对比

| Task | Baseline / vanilla | CSD 表现 | 主要现象 |
|---|---:|---:|---|
| GSM8K | baseline ≈ 0.92~0.94 | CSD ≈ 0.89~0.95 | 不是全局坏掉，CSD 有时略好 |
| AIME25 | baseline/vanilla ≈ 0.90~0.93 | CSD ≈ 0.80~0.93 | 小样本波动大，CSD 均值略低 |
| Math500 | baseline/vanilla ≈ 0.824~0.854 | CSD ≈ 0.812~0.858 | 静态接近 baseline，dynamic 更差 |
| LCB codegeneration | baseline ≈ 0.79 或 0.68，取决于生成参数 | CSD 常降到 0.72、0.65 或更低 | 对 force accept 最敏感 |

所以更准确的说法不是“CSD 整体不可用”，而是：

> CSD 的问题具有任务选择性。GSM8K/Math500 上静态 CSD 可以接近甚至偶尔超过 baseline；AIME25 有小样本波动但总体还能跑；LCB/code-generation 对错误 force accept 最敏感，因此暴露出最严重的精度问题。

这也进一步支持：问题不是单纯的 kernel 全局错误，而是 force-accept 策略在高熵、长程依赖、格式敏感生成中的风险。

## 5. 更大的 ratio 不代表精度单调更好

table 干净程度随 ratio 增大而单调提升：

```text
ratio0.01 table > ratio0.3 table > ratio1 table
```

但任务精度并不单调提升。

| Table | LCB accuracy example |
|---|---:|
| logits-gated ratio0.01 table | 0.708571~0.714286 |
| logits-gated ratio0.3 table | 0.731429~0.737143 |
| logits-gated ratio1 table | 0.702857~0.731429 |

所以 ratio1 更干净，但不一定更准。

解释：

- `ratio=1` 意味着 draft token 必须在当前 gate 下与 target max-logit token 打平或相等。
- 这是一个严格的局部条件，但局部 top-1/top-tie 不等于全局语义安全。
- 在 LCB / CoT 中，高熵分支点和格式敏感位置会让局部看起来合理的 force accept 变成有害替换。

## 6. Dynamic update 明显更危险

多组 LCB 结果中，dynamic CSD 都比 static CSD 更差。

| Setting | Static CSD | Dynamic CSD |
|---|---:|---:|
| old table, steps5/topk3/draft15, LCB | 0.720000 | 0.645714 |
| old table, steps5/topk1/draft5, LCB | 0.714286 | 0.645714 |
| low-ratio table, LCB | 0.691429 | 0.434286 |
| ratio1 table, steps5/topk3/draft15, LCB | 0.702857 | 0.651429 |
| ratio0.3 table, steps3/topk1/draft3, LCB | 0.737143 | 0.628571 |

结论：

> Static CSD 在 LCB 上已经有精度风险；dynamic update 会进一步放大这个风险。

一个可能原因是：online update 会加入一些在当前 target logits 下局部合理的 pair，但这些 pair 对长程推理和代码生成未必安全。

## 7. 当前工作解释

当前最合理的解释是：

1. **Table cleaning 是真实有效的。**  
   Logits-gated calibration 和更小的 speculative shape 都显著减少了 table 大小。

2. **但 table 干净程度还不够。**  
   即使 ratio1 table 也无法恢复 LCB baseline 精度。

3. **LCB/CoT/code generation 比 GSM8K/lm_eval 更敏感。**  
   CSD 在 lm_eval GSM8K 上可以正常甚至略有收益，但 LCB 明显更脆弱。

4. **真正危险的是 force-accept 决策。**  
   当前 CSD 依赖局部 token/logit 条件和 table membership 做 accept，这不足以保证长程推理和代码生成的下游正确性。

5. **Dynamic update 暂时应该视为高风险。**  
   现有结果反复显示 dynamic CSD 在 LCB 上比 static CSD 更差。

## 8. 下一步 debug 方向

下一步最有价值的证据应该来自 LCB 上的 force-accept event logging。最重要的字段包括：

- `event_flag`
- `spec_step`
- `draft_prob`
- `resampled_prob`
- `max_prob`
- `target_entropy`
- `normalized_entropy`
- `logit_margin_to_max`
- `pair_logit_margin`
- `(lhs_token, rhs_token)` pair 分布

关键问题：

1. 有害 force accept 是否集中在高 target entropy 位置？
2. 是否集中在更深的 speculative step？
3. ratio1 在 LCB/code-generation 位置上是否仍然 force accept 过多？
4. dynamic-update pair 是否比 static table pair 质量更低？
5. 某些 token 类别是否在坏的 force accept 中过度出现，例如空格、标点、代码符号、缩进、reasoning marker？

## 最终结论

当前证据支持以下结论：

> 我们确实构建出了更干净的 CSD table，但更干净的 table 本身并不能修复 LCB 精度。剩下的问题更可能是高熵长程生成中的 force-accept 策略问题，而不只是 table 太脏。
