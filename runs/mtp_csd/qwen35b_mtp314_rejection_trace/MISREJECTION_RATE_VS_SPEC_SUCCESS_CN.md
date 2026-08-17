# 误拒率、级联收益与投机成功率的关系

## 1. 本文要回答的问题

本文区分三个容易混淆的量：拒绝事件数、误拒率和投机成功率。核心结论是：一次 CSD force accept 不仅可以恢复当前被误拒的 draft token，还可能让同一轮验证继续接受后续 draft token。因此，拒绝事件数即使基本不变，投机成功率仍然可以显著提高；误拒率也不能直接与投机成功率的百分点增量画等号。

本文讨论当前 Qwen3.5 MTP `steps=3, topk=1` 的链式验证。为避免混淆，所有公式中的“接受数”只计算 draft token，不包含每轮额外提交的 bonus/residual token。

## 2. 指标定义

设：

- $Q$：speculative verify round 的数量；
- $K$：每轮可验证的 draft 槽位数；当前配置中 $K=3$；
- $a_i\in\{0,1,\ldots,K\}$：第 $i$ 轮实际接受的 draft token 数；
- $A=\sum_{i=1}^{Q}a_i$：所有轮累计接受的 draft token 数；
- $R$：记录到首个拒绝的 verify round 数；全接受轮没有 rejection event，因此 $R\le Q$；
- $M$：被判定为误拒、并适合恢复的 rejection event 数；
- $m=M/R$：以 rejection event 为分母的误拒率；
- $g_e$：恢复误拒事件 $e$ 后，该轮相对 Bare 新增的 draft 接受数。

### 2.1 平均接受长度

每个 verify round 除 draft token 外还会提交一个 bonus/residual token。因此仓库中的平均接受长度为：

$$
L=1+\frac{A}{Q}.
$$

这里的常数 $1$ 就是每轮至少产生的 bonus/residual token。

### 2.2 投机成功率

仓库当前的聚合实现为：

$$
S=\frac{L-1}{K}=\frac{A}{KQ}.
$$

对应代码口径为：

```text
aggregate_accept_length = total_completion_tokens / total_spec_verify_ct
spec_success_rate = (aggregate_accept_length - 1) / speculative_num_steps
```

因此，投机成功率衡量的是全部 draft 槽位中有多少最终被接受，而不是“有多少 verify round 完全没有发生拒绝”。

### 2.3 误拒率

当前 rejection trace 在每个 topk=1 MTP chain 中只记录首个拒绝。因此：

$$
m=\frac{M}{R}
$$

回答的是：

> 在已经产生首拒事件的 verify round 中，有多少首拒边界可能是不必要的？

它并不回答：

> 所有 draft 槽位中有多少 token 可以被额外接受？

两个指标的分母不同。

## 3. 从 70% 成功率和 20% 误拒率推导新成功率

“当前投机成功率为 70%，失败为 30%，其中 20% 是误拒”有两种不同含义。只有先确定 20% 的分母，才能计算 CSD 最终可以提高多少投机成功率。

### 3.1 20% 是失败 draft 槽位中的误拒比例

设 Bare 投机成功率为：

$$
S=70\%.
$$

Bare 中没有接受的 draft 槽位比例为：

$$
1-S=30\%.
$$

设这些失败槽位中可恢复的比例为 $\rho=20\%$。如果全部恢复，新增成功比例为：

$$
\Delta S=(1-S)\rho=30\%\times20\%=6\%.
$$

因此新的投机成功率为：

$$
S_{\mathrm{new}}=S+(1-S)\rho=70\%+6\%=76\%.
$$

此时可以从三个角度描述收益：

| 口径 | 结果 |
| --- | ---: |
| 投机成功率绝对提升 | 6 个百分点 |
| 原失败槽位减少比例 | 20% |
| 成功率相对增长 | $6/70=8.57\%$ |

这里的 $\rho$ 已经是 token 槽位级的最终可恢复比例。直接恢复和后续级联恢复都应当先计入“可恢复槽位数”，再计算 $\rho$。如果已经给定 $\rho=20\%$，就不能再额外乘一次级联长度，否则会重复计算同一批 token 收益。

一般公式为：

$$
S_{\mathrm{new}}=S+(1-S)\rho.
$$

### 3.2 20% 是首拒事件中的误拒比例

当前 judge 更接近事件级定义：

$$
m=\frac{M}{R}=20\%.
$$

这表示 20% 的首拒事件被判为可以恢复，但没有直接说明每个事件最终能恢复多少个失败 draft 槽位。此时必须引入：

- $r=R/Q$：产生首拒事件的 verify rounds 比例；
- $\bar c$：第一次 force accept 后，平均额外正常接受或再次 force accept 的后续 token 数；
- $\bar g=1+\bar c$：每个误拒事件平均恢复的总 draft 数，其中常数 1 是当前被 force accept 的 token。

事件级误拒带来的投机成功率提升为：

$$
\Delta S=m\cdot r\cdot\frac{\bar g}{K}
=m\cdot r\cdot\frac{1+\bar c}{K}.
$$

新投机成功率为：

$$
S_{\mathrm{new}}
=S+m\cdot r\cdot\frac{1+\bar c}{K}.
$$

转换为“失败槽位中实际恢复了多少”的 token 级比例：

$$
\rho
=\frac{\Delta S}{1-S}
=\frac{m\,r\,(1+\bar c)}{K(1-S)}.
$$

### 3.3 不同平均级联长度的数值分析

使用下面的直观示例：

```text
Bare 投机成功率 S = 70%
事件级误拒率 m = 20%
每轮 draft 槽位 K = 3
产生首拒事件的轮次比例 r = R/Q = 50%
```

这里 $r=50\%$ 是便于理解的示例值，也接近当前三个 trace 中约 44%–56% 的范围。不同平均后续级联长度 $\bar c$ 对应的结果如下。

| 平均后续级联 $\bar c$ | 平均总恢复 $\bar g=1+\bar c$ | 成功率提升 $\Delta S$ | 新成功率 | 原失败槽位恢复比例 $\rho$ |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1.0 | 3.33 个百分点 | 73.33% | 11.11% |
| 0.5 | 1.5 | 5.00 个百分点 | 75.00% | 16.67% |
| 1.0 | 2.0 | 6.67 个百分点 | 76.67% | 22.22% |
| 1.5 | 2.5 | 8.33 个百分点 | 78.33% | 27.78% |
| 2.0 | 3.0 | 10.00 个百分点 | 80.00% | 33.33% |

表中的第一行表示没有后续级联：每个误拒事件只恢复当前 token。最后一行表示 `steps=3` 下的最大平均收益：误拒发生在第一个 draft 位置，而且当前 token 与后续两个 draft 全部被接受。

例如平均后续级联为 $\bar c=1$ 时，每个误拒事件平均恢复两个 draft token：

$$
\Delta S
=0.20\times0.50\times\frac{2}{3}
=6.67\%.
$$

因此新成功率为：

$$
S_{\mathrm{new}}=70\%+6.67\%=76.67\%.
$$

### 3.4 两种 20% 为什么会得到不同结果

如果“20%误拒”已经指失败 token 槽位中的20%，则：

$$
S_{\mathrm{new}}=76\%.
$$

如果“20%误拒”指首拒事件中的20%，则结果取决于 $r$ 与 $\bar g$。在上面的 $r=50\%, K=3$ 示例中：

- 无后续级联时，新成功率为 73.33%；
- 平均再接受 1 个后续 token 时，新成功率为 76.67%；
- 平均再接受 2 个后续 token 时，新成功率为 80.00%。

因此，若希望把 judge 结果与投机成功率直接比较，必须先将事件级误拒率转换成 token 加权恢复率：

$$
\rho=\frac{\sum_{e=1}^{M}g_e}{KQ(1-S)}.
$$

转换后再使用：

$$
S_{\mathrm{new}}=S+(1-S)\rho.
$$

## 4. 三个请求的简单例子

假设有三个请求，每个请求当前都执行一个 verify round，每轮提出三个 draft token，因此：

$$
Q=3,\qquad K=3,\qquad KQ=9.
$$

符号约定如下：

- `✓`：正常验证接受；
- `✗`：正常验证拒绝；
- `F`：原本会拒绝，但被 CSD force accept；
- `—`：Bare 在更早位置停止，因此该 token 没有沿有效路径继续接受。

### 3.1 Bare MTP

| 请求 | Bare 验证路径 | 接受 draft 数 | 首拒位置 |
| --- | --- | ---: | ---: |
| 请求 1 | `✗ — —` | 0 | 1 |
| 请求 2 | `✓ ✗ —` | 1 | 2 |
| 请求 3 | `✓ ✓ ✗` | 2 | 3 |

总接受数为：

$$
A_{\mathrm{bare}}=0+1+2=3.
$$

Bare 投机成功率为：

$$
S_{\mathrm{bare}}=\frac{3}{3\times3}=\frac{1}{3}=33.3\%.
$$

三个请求都产生了首拒事件，因此 $R=3$。假设只有请求 1 的首拒属于误拒，则：

$$
M=1,\qquad m=\frac{1}{3}=33.3\%.
$$

### 3.2 CSD 只恢复当前 token，没有级联

如果请求 1 的第一个 draft 被 force accept，但第二个 draft 随即正常拒绝，则：

| 请求 | CSD 验证路径 | 接受 draft 数 | 相对 Bare 增量 |
| --- | --- | ---: | ---: |
| 请求 1 | `F ✗ —` | 1 | +1 |
| 请求 2 | `✓ ✗ —` | 1 | 0 |
| 请求 3 | `✓ ✓ ✗` | 2 | 0 |

此时一次误拒恢复的收益为 $g_1=1$，总接受数变为：

$$
A_{\mathrm{csd}}=1+1+2=4.
$$

投机成功率变为：

$$
S_{\mathrm{csd}}=\frac{4}{9}=44.4\%.
$$

绝对提升为：

$$
\Delta S=44.4\%-33.3\%=11.1\%.
$$

虽然误拒率是 $33.3\%$，投机成功率只提高了 $11.1$ 个百分点，因为误拒率以 3 个 rejection events 为分母，而投机成功率以 9 个 draft 槽位为分母。

### 3.3 CSD 恢复当前 token，并产生级联

如果请求 1 的第一个 draft 被 force accept 后，第二个 draft 也正常通过，最后在第三个 draft 才发生真正拒绝，则：

| 请求 | CSD 验证路径 | 接受 draft 数 | 相对 Bare 增量 |
| --- | --- | ---: | ---: |
| 请求 1 | `F ✓ ✗` | 2 | +2 |
| 请求 2 | `✓ ✗ —` | 1 | 0 |
| 请求 3 | `✓ ✓ ✗` | 2 | 0 |

请求 1 的一次误拒恢复包含两部分：

```text
直接收益：第1个 draft 被 force accept，+1
级联收益：第2个 draft 得以继续并正常接受，+1
总收益：g_1 = 2
```

总接受数变为：

$$
A_{\mathrm{csd}}=2+1+2=5,
$$

投机成功率变为：

$$
S_{\mathrm{csd}}=\frac{5}{9}=55.6\%,
$$

绝对提升为：

$$
\Delta S=55.6\%-33.3\%=22.2\%.
$$

此时三个请求仍然各有一个终止拒绝，拒绝事件数仍为 3；请求 1 的拒绝只是从第 1 个位置移动到了第 3 个位置。投机成功率提高来自每轮接受的 draft token 数增加，而不是必须消灭该轮的最终拒绝事件。

## 5. 固定 verify rounds 下的一般公式

为了得到可解释的因果分解，先考虑 Bare 与 CSD 使用完全相同的 $Q$ 个 verify rounds、相同 candidate tree 和相同采样随机数的 matched counterfactual。对每个被恢复的误拒事件 $e$，定义：

$$
g_e=a_e^{\mathrm{csd}}-a_e^{\mathrm{bare}}.
$$

这里 $g_e\ge1$，并且可以分解为：

$$
g_e=1+c_e+f_e,
$$

其中：

- $1$：当前误拒 token 的直接 force-accept 收益；
- $c_e$：恢复路径后，后续正常验证接受的 token 数；
- $f_e$：同一轮后续再次发生 force accept 所带来的额外 token 数。

在固定 $Q$ 的反事实条件下：

$$
\Delta S
=\frac{\sum_{e=1}^{M}g_e}{KQ}.
$$

令误拒事件的平均总收益为：

$$
\bar g=\frac{1}{M}\sum_{e=1}^{M}g_e,
$$

则：

$$
\Delta S=\frac{M\bar g}{KQ}.
$$

又因为 $M=mR$，所以：

$$
\Delta S=m\cdot\frac{R}{Q}\cdot\frac{\bar g}{K}
$$

这个公式给出了误拒率与投机成功率提升之间最重要的关系：

- $m$ 越高，可恢复的首拒事件越多；
- $R/Q$ 表示 verify rounds 中实际产生首拒事件的比例；
- $\bar g$ 表示每次恢复平均新增多少个接受 token；
- $K$ 是每轮总 draft 槽位数。

因此，即使误拒率 $m$ 不变，更大的级联收益 $\bar g$ 仍会显著提高投机成功率。

### 4.1 无级联的特殊情况

如果每次误拒恢复只接受当前 token，后面立即拒绝，则 $\bar g=1$：

$$
\Delta S=m\cdot\frac{R}{Q}\cdot\frac{1}{K}.
$$

这时误拒率对投机成功率的贡献被 $R/Q$ 和 $1/K$ 同时缩放。

### 4.2 最大级联的上界

对于链长 $K$，若误拒发生在位置 $p_e\in\{1,\ldots,K\}$，则单次事件最多新增：

$$
g_e\le K-p_e+1.
$$

发生得越早的误拒，潜在级联空间越大。例如 $K=3$ 时：

| 误拒位置 | 最大新增接受数 |
| ---: | ---: |
| 1 | 3 |
| 2 | 2 |
| 3 | 1 |

因此，只报告误拒事件数量仍然不够；还必须报告误拒位置分布和位置恢复后的真实级联长度。

## 6. 当前 trace 的真实拒绝覆盖率

当前大规模 trace 中，每个 verify round 最多记录一个首拒事件。已有数据为：

| 任务 | Verify rounds $Q$ | Rejection events $R$ | $R/Q$ |
| --- | ---: | ---: | ---: |
| LCB v6 | 381,743 | 212,028 | 55.54% |
| AIME 2025 | 231,195 | 101,511 | 43.91% |
| OlympiadBench EN | 320,362 | 148,908 | 46.48% |

这些数据说明，并非每轮都有 rejection event。约 44%–56% 的 verify rounds 产生了首拒，其余轮已经接受了全部 draft token。因此，将误拒率直接视为“所有 draft 槽位中可新增接受的比例”会重复高估或低估不同因素。

以 LCB 为例，若暂时使用 $m=19\%$、$R/Q=55.54\%$、$K=3$，则：

$$
\Delta S
=0.19\times0.5554\times\frac{\bar g}{3}.
$$

若没有级联，$\bar g=1$：

$$
\Delta S\approx3.52\%.
$$

若平均每次恢复新增两个 token，$\bar g=2$：

$$
\Delta S\approx7.04\%.
$$

若每次都达到 $K=3$ 的理论最大收益：

$$
\Delta S\approx10.55\%.
$$

这里的 $19\%$ 只是当前 judge 的阶段性宽口径比例，而且 trace 子集与正式 CSD 全量运行并非 matched counterfactual。因此上面的数字只能展示公式量级，不能用于宣称已经精确解释了正式结果。

## 7. 为什么端到端两次运行不能直接套用固定分母公式

正式 Bare 与 CSD 运行中，CSD 每轮提交更多 token，会减少完成相同输出所需的 verify rounds。此外，temperature 为 1 时，force accept 会改变后续上下文和随机生成轨迹，因此两种方法通常满足：

$$
Q_{\mathrm{bare}}\ne Q_{\mathrm{csd}}.
$$

两次端到端运行的成功率分别为：

$$
S_{\mathrm{bare}}
=\frac{A_{\mathrm{bare}}}{KQ_{\mathrm{bare}}},
\qquad
S_{\mathrm{csd}}
=\frac{A_{\mathrm{csd}}}{KQ_{\mathrm{csd}}}.
$$

此时：

$$
S_{\mathrm{csd}}-S_{\mathrm{bare}}
$$

同时包含：

1. 当前误拒 token 的直接恢复；
2. 同一轮后续 token 的级联接受；
3. 同一轮可能发生的多次 force accept；
4. 每轮提交 token 增加导致的 verify-round 数变化；
5. 生成轨迹、输出长度和上下文分布变化；
6. Dynamic 模式下 table 随运行更新产生的额外变化。

因此，使用：

$$
\frac{S_{\mathrm{csd}}-S_{\mathrm{bare}}}{1-S_{\mathrm{bare}}}
$$

只能得到“表观拒绝恢复率”，不能把它解释成严格的误拒事件恢复比例。

## 8. 新 trace 应记录的字段

为了将 judge 的误拒判定与系统接受率提升一一对应，新 trace 应在真实 CSD verifier 路径上记录：

```json
{
  "accepted_drafts_before_first_force": 0,
  "first_force_position": 1,
  "forced_accept_count_in_chain": 1,
  "normal_accept_count_after_first_force": 1,
  "final_accepted_drafts": 2,
  "cascade_gain": 2,
  "terminal_rejection_position": 3
}
```

字段定义如下：

- `accepted_drafts_before_first_force`：第一次 force accept 前已经正常接受的 draft 数；
- `first_force_position`：第一次被 CSD 恢复的位置；
- `forced_accept_count_in_chain`：本轮 CSD 直接 force accept 的 token 数；
- `normal_accept_count_after_first_force`：第一次 force 后，后续正常接受的 token 数；
- `final_accepted_drafts`：本轮最终接受的全部 draft 数；
- `cascade_gain`：相对于在第一次 force 位置立即停止，新增提交的 draft 数；
- `terminal_rejection_position`：恢复路径后最终停止的位置；若全部 draft 接受则为 `null`。

最核心的量为：

```text
cascade_gain
= final_accepted_drafts
- accepted_drafts_before_first_force
```

进一步可将其拆为：

```text
cascade_gain
= forced_accept_count_in_chain
+ normal_accept_count_after_first_force
```

如果同一轮可能多次 force accept，还应保留完整位置列表，而不是只记录计数：

```json
"force_positions": [1, 3]
```

## 9. 推荐报告口径

最终报告应同时给出以下四组指标：

| 指标 | 回答的问题 |
| --- | --- |
| Judge 局部可替换率 $M/R$ | 首拒事件中有多少具有局部语义或功能依据？ |
| CSD 实际 force-accept 率 | CSD 实际干预了多少首拒边界？ |
| 平均级联收益 $\bar g$ | 每次干预平均新增多少个接受 token？ |
| 端到端投机成功率 $S$ | 所有 draft 槽位中最终接受了多少？ |

推荐将系统收益分解为：

```text
直接 force-accept 收益
+ force 后正常接受的级联收益
+ 同轮后续再次 force 的收益
= 总新增 draft 接受数
```

精度和吞吐仍需单独报告，因为更高的接受率不自动保证任务精度不变，也不自动保证 verifier 与动态更新开销能够被新增接受 token 完全抵消。

## 10. 结论与证据边界

误拒率描述的是 rejection-event 层面的可恢复机会，投机成功率描述的是 draft-slot 层面的最终接受比例。二者通过 $R/Q$、平均级联收益 $\bar g$ 和 draft 长度 $K$ 联系：

$$
\Delta S
=m\cdot\frac{R}{Q}\cdot\frac{\bar g}{K}.
$$

三请求例子说明，即使每个请求最终仍然发生一次拒绝，一次误拒恢复也可以把拒绝位置后移，并让投机成功率提高多个 token 槽位。当前 trace 只记录首拒边界，尚未记录恢复后的级联长度，因此不能仅依靠 judge 比例解释端到端投机成功率增量。补充真实 CSD 路径上的级联字段，是验证这一因果关系的必要下一步。

## 11. 自检与主张—证据对应

### 10.1 自检

- 清晰性：误拒率、拒绝事件、draft 槽位、接受长度和投机成功率已分别定义。
- 公式边界：固定 $Q$ 的反事实公式与端到端不同 $Q$ 的比较已明确区分。
- 术语一致性：`cascade_gain` 始终表示第一次 force 后相对原停止位置新增的 draft 接受数。
- 未支持主张：本文没有宣称当前 judge 比例已经精确解释正式 CSD 结果。
- 缺失证据：尚缺 per-event 真实级联 trace，需新增 instrumentation 后验证。

### 10.2 主张—证据映射

| 主张 | 证据 | 状态 |
| --- | --- | --- |
| 拒绝事件数不变时，投机成功率仍可提高 | 三请求例子中拒绝事件保持 3 个，接受数从 3 增至 5 | 已支持 |
| 误拒率不能直接等同于投机成功率百分点增量 | 两者分母分别为 $R$ 与 $KQ$，一般公式显式包含 $R/Q$、$\bar g$ 和 $K$ | 已支持 |
| 级联可能解释一部分系统接受率增量 | verifier 在 force accept 后继续验证后续 draft；公式给出放大机制 | 机制已支持，量级待测 |
| 当前系统增量主要来自级联 | 当前 trace 尚无 per-event `cascade_gain` | 需要新证据 |
