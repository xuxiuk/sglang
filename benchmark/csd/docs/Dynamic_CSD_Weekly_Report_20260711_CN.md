# Dynamic CSD 周报实验汇总（2026-07-11）

## 1. 本周目标

本周围绕 Dynamic CSD 在 coding OOD 数据集上“投机成功率提升明显，但吞吐提升未完全同比转化”的问题，完成了 CPU rebuild 开销定位、stateful C++ rebuild 实现、长上下文 verify 成本验证，以及 APPS、TACO、LiveCodeBench（LCB）长输出实验。

本周得到的核心结论是：

1. 原 Python rebuild 即使放入 `ThreadPoolExecutor`，仍会因 GIL 抢占 scheduler 的 Python 执行时间。
2. Stateful C++ builder 将频次状态、增量更新和 CPU hash-table build 移入 C++，消除了主要 rebuild GIL 热点。
3. Dynamic CSD 的剩余吞吐损失主要不能再归因于 rebuild；更高接受率会改变输出长度和运行上下文分布，长上下文下 target verify 的 FlashAttention 成本随之上升。
4. 在最新 coding OOD 全量采样中，Dynamic 相对 Plain 提升 7.84% 至 12.95%，相对原始 Eagle 提升 31.63% 至 38.70%。

## 2. 实现状态

新增实现位于：

- `sgl-kernel/csrc/speculative/csd_rebuild.cpp`
- `python/sglang/srt/speculative/csd_runtime.py`
- `sgl-kernel/tests/speculative/test_csd_rebuild.py`

`CSDTableBuilder` 在 C++ 中长期维护：

- 全量 token-pair frequency map；
- 达到 frequency threshold 的 active-key set；
- open-addressing hash table 构建状态。

在线阶段只将本轮 delta pairs 传入 C++，调用 `update_and_build()` 后返回 CPU hash-table Tensor。Python 负责后台 future 生命周期以及最终 CPU-to-GPU materialize/swap。

源码已经加入 `sgl-kernel/CMakeLists.txt`。当前实验通过独立编译的 `benchmark/csd/native/csd_rebuild_cpu_ext.so` 加载同一份 C++ 实现；正式部署需要重新构建并安装 `sgl-kernel`，之后不再需要 `SGLANG_CSD_NATIVE_EXTENSION_PATH`。

## 3. 最新实验配置

模型与 speculative 配置：

- tree：`515:5:1:5`
- temperature：1.0
- top-p：0.95
- top-k：20
- presence penalty：1.5
- batch parallel：24
- CSD rebuild threshold：4096
- 方法：Eagle、Plain、Dynamic、Dynamic + entropy P20

数据配置：

| 数据集 | 样本数 | max new tokens |
| --- | ---: | ---: |
| APPS | 1000 | 40,960 |
| TACO | 1000 | 40,960 |
| LCB v6 | 175 | 81,920 |

这里的 40,960/81,920 是生成输出上限，不是固定的实际 attention context。实际 verify context 由 prompt、已生成 token 和 batch 调度共同决定。

## 4. 最新吞吐结果

单位均为 output token/s。

| 数据集 | Eagle | Plain | Dynamic | Dynamic + P20 |
| --- | ---: | ---: | ---: | ---: |
| APPS | 2895.7 | 3535.6 | **3993.0** | 3738.4 |
| TACO | 2847.7 | 3497.0 | **3949.7** | 3698.4 |
| LCB | 2474.7 | 3020.9 | **3257.6** | 3241.6 |

相对 Plain：

| 数据集 | Dynamic | Dynamic + P20 |
| --- | ---: | ---: |
| APPS | **+12.94%** | +5.74% |
| TACO | **+12.95%** | +5.76% |
| LCB | **+7.84%** | +7.31% |
| 简单平均 | **+11.24%** | +6.27% |

相对 Eagle：

| 数据集 | Plain | Dynamic | Dynamic + P20 |
| --- | ---: | ---: | ---: |
| APPS | +22.10% | **+37.89%** | +29.10% |
| TACO | +22.80% | **+38.70%** | +29.87% |
| LCB | +22.07% | **+31.63%** | +30.99% |

## 5. 投机接受结果

| 数据集 | 方法 | accept length | speculative token ratio |
| --- | --- | ---: | ---: |
| APPS | Eagle | 3.366 | 47.32% |
| APPS | Plain | 4.070 | 61.40% |
| APPS | Dynamic | **4.632** | **72.64%** |
| APPS | Dynamic + P20 | 4.372 | 67.44% |
| TACO | Eagle | 3.313 | 46.26% |
| TACO | Plain | 4.007 | 60.14% |
| TACO | Dynamic | **4.511** | **70.22%** |
| TACO | Dynamic + P20 | 4.237 | 64.74% |
| LCB | Eagle | 3.359 | 47.18% |
| LCB | Plain | 4.072 | 61.44% |
| LCB | Dynamic | **4.489** | **69.78%** |
| LCB | Dynamic + P20 | 4.158 | 63.16% |

Dynamic 在三个数据集上均取得最高接受长度和 speculative-token ratio。P20 限制减少了 replacement 覆盖，APPS/TACO 上吞吐也随接受率同步回落；当前 P20 更像保守策略，而不是最佳性能策略。

## 6. 与上一轮 81,920 实验对比

### 6.1 81,920 汇总结果（允许跨轮引用）

为了在周报中保留完整参考，下面将 81,920 的结果统一列出。标记 `S` 的数据来自 7 月 10 日 Stateful C++ 轮；标记 `H` 的数据来自 7 月 8 日历史轮。历史轮使用相同模型、tree、temperature、top-p、top-k、presence penalty、parallel 和数据集，但不是同一次启动，因此只能用于趋势参考，不能当作严格 A/B。

| 数据集 | Eagle | Plain | Dynamic | Dynamic + P20 |
| --- | ---: | ---: | ---: | ---: |
| APPS | 2711.7 `S` | 3416.4 `H` | 3584.9 `H` | 3364.6 `H` |
| TACO | 2696.3 `S` | 3485.7 `H` | 3741.1 `S` | 3574.0 `S` |
| LCB | 2436.0 `S` | 3092.6 `H` | 3245.7 `S` | 3052.8 `S` |

对应 accept length：

| 数据集 | Eagle | Plain | Dynamic | Dynamic + P20 |
| --- | ---: | ---: | ---: | ---: |
| APPS | 3.346 `S` | 4.116 `H` | 4.944 `H` | 4.524 `H` |
| TACO | 3.331 `S` | 4.077 `H` | 4.689 `S` | 4.397 `S` |
| LCB | 3.361 `S` | 4.147 `H` | 4.434 `S` | 4.131 `S` |

若用跨轮 Plain 作为参考，81,920 下的吞吐变化为：

| 数据集 | Dynamic vs Plain | Dynamic + P20 vs Plain |
| --- | ---: | ---: |
| APPS | +4.93% | -1.52% |
| TACO | +7.33% | +2.53% |
| LCB | +4.95% | -1.29% |

这组跨轮数据进一步说明，81,920 极长输出配置下 Dynamic 的吞吐转化明显弱于最新 40K 配置；但由于 Plain 来自另一轮，绝对差值仍需以统一 32K 新实验为准。

原始数据来源：

- Stateful C++：`benchmark/csd/runs/20260710_stateful_cpp_full_apps/`、`benchmark/csd/runs/20260710_stateful_cpp_full_taco_lcb/`
- 历史 Plain/APPS Dynamic：`benchmark/csd/runs/domain_ood_csd_515/20260708_code_ood_huge_g1/`

### 6.2 TACO 81,920 对 40,960

上一轮 TACO 使用 `max_new_tokens=81920`，本轮使用 40,960。除 Plain 外，下面是同一 Stateful C++ 实现的对比：

| 方法 | TACO 81,920 | TACO 40,960 | 变化 |
| --- | ---: | ---: | ---: |
| Eagle | 2696.3 | 2847.7 | **+5.62%** |
| Dynamic | 3741.1 | 3949.7 | **+5.57%** |
| Dynamic + P20 | 3574.0 | 3698.4 | **+3.48%** |

因此，“40K 比之前 80K 吞吐提高不少”在 TACO 上成立。Eagle 和 Dynamic 都提高约 5.6%，说明主要收益来自减少极长生成和长 context verify，而不是只对 Dynamic 生效。

### 6.3 LCB 重复实验

LCB 上一轮和本轮都使用 81,920，只是重复测量：

| 方法 | 上一轮 | 本轮 | 变化 |
| --- | ---: | ---: | ---: |
| Eagle | 2436.0 | 2474.7 | +1.59% |
| Dynamic | 3245.7 | 3257.6 | +0.37% |
| Dynamic + P20 | 3052.8 | 3241.6 | +6.18% |

LCB 的变化不能归因于输出上限缩短。P20 波动较大，且只有 175 条样本，需要结合重复实验和输出长度分布判断。

APPS 的 81,920 四方法数据来自两轮拼接，因此也不能给出严格的 81,920 对 40,960 A/B，只在 6.1 中作为趋势参考。

## 7. 为什么接受率没有等比例转化为吞吐

吞吐是整条 decode pipeline 的摊销结果，不只由每次 speculative step 接受多少 token 决定。主要因素包括：

1. target verify 的 attention 成本随上下文长度增加；
2. Dynamic 可能改变 EOS/max-new 命中率和实际输出长度分布；
3. 长请求造成 batch tail，后期有效 batch 变小；
4. prefill、draft、verify、post-verify、scheduler 和 CPU/GPU 同步均计入端到端吞吐；
5. LCB 长尾尤其明显，本轮 Dynamic 的 tail time ratio 为 21.8%。

受控 profile 已验证：固定输出长度但增加 verify context 时，target FlashAttention 开销显著上升；在相同 context 下，Dynamic 不比 Eagle/Plain 慢。这支持“剩余差距主要来自 workload/context 改变”，而不是新的 C++ rebuild 或 lookup 本身变慢。

## 8. 当前判断

- 最佳纯性能配置：Dynamic，无 P20。
- APPS/TACO：Dynamic 相对 Plain 稳定提升约 13%。
- LCB：Dynamic 仍领先，但长上下文和长尾将增益压缩到约 8%。
- P20：接受率和吞吐均低于 unrestricted Dynamic；是否保留应由最终代码正确率决定。
- 代码精度：原始 outputs 已保存，但执行式 accuracy evaluator 尚未完成，因此当前不能用吞吐结果替代 pass rate 结论。

## 9. 下一步实验

下一轮将 APPS、TACO、LCB 统一设置为 `max_new_tokens=32768`，样本数和四方法保持不变。统一上限可减少 workload 分布差异，并用于回答：

1. 从 40K/80K 降到 32K 后吞吐是否继续提高；
2. Dynamic 相对 Plain 的提升是否更接近接受率带来的理论收益；
3. LCB 长尾是否显著收敛；
4. P20 与 unrestricted Dynamic 的差距是否仍然存在。

原始模型输出仅本地保存用于精度评估，不提交 Git；提交内容只保留 summary、result JSONL、request metrics、completed windows 和必要日志。
