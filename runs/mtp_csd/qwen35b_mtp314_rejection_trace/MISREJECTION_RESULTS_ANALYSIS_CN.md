# MTP 投机解码误拒结果分析

## 1. 结论摘要

本实验最终采用 **Direct Local Judge V5.5.1** 估计首拒位置的局部误拒候选率。它判断的不是
“draft 分支最终能否答对”，而是：在完整共享上下文已经固定的前提下，将 verifier 选择的
residual token 换成被拒绝的 draft token，是否保持当前边界已经建立的任务相关语义、操作或
行为。

最终结果如下：

| 指标 | 结果 |
| --- | ---: |
| 完整 trace 原始拒绝事件 | 462,447 |
| 去重后可抽样事件 | 462,441 |
| 全局无放回抽样比例 | 5% |
| 请求 Judge 的事件 | 23,122 |
| 成功获得合法判断 | 22,051（95.37%） |
| 永久失败并跳过 | 1,071（4.63%） |
| `LOCALLY_SUBSTITUTABLE` | 4,334 |
| 首拒事件级局部误拒候选率 | **19.65%** |
| 近似 95% 置信区间 | **19.13%–20.18%** |

后续随机案例审计发现 `LOCALLY_SUBSTITUTABLE` 中存在明显假阳性，例如把数学对象
`circle/Disk`、概念 `number/carry`、LaTeX token `\\/<` 和不同 Python 缩进判为可替换。
因此 19.65% 应严格称为 **Direct-local Judge candidate rate**，而不是人工验证后的真实误拒率；
上面的二项区间也只描述 Judge 标签的抽样波动，不包含 Judge 系统误差。

最重要的解释是：这里的 19.65% 以“被记录的首拒事件”为分母，不以“所有失败 draft
槽位”为分母。因此它不能直接代入

$$
S_{\mathrm{new}}=S+(1-S)m
$$

把 70% 成功率直接换算成 75.90%。要推导投机成功率提升，还必须知道发生首拒的 verify
round 比例和每次纠正误拒后恢复的平均级联长度。

## 2. 最终实验口径

### 2.1 数据来源

模型和投机配置为 Qwen3.5-35B-A3B 与 MTP 3-1-4。Trace 来自三个任务：

```text
runs/mtp_csd/qwen35b_mtp314_rejection_trace/runs/
  trace_large_80_30_80_20260814_001505/
    lcb_v6/
    aime25/
    olympiad_math_en/
```

原始事件数为：

| 任务 | 首拒事件 |
| --- | ---: |
| LiveCodeBench v6 | 212,028 |
| AIME 2025 | 101,511 |
| OlympiadBench Math EN | 148,908 |
| 合计 | 462,447 |

按事件标识去重后有 462,441 条；全局按 `seed=42` 无放回抽取 5%，得到 23,122 条。抽样时
不读取 calibration frequency、table hit、probability ratio、entropy 或 Judge 标签，因而
不会预先偏向高频或低频事件。

### 2.2 每条样本包含什么

Judge 看到：

```text
完整原始 prompt
+ 拒绝位置之前的全部生成 token
+ 盲化后的 Candidate A 与 Candidate B
```

候选 token 后没有续写。A/B 顺序被确定性随机化，Judge 不知道哪一个是 draft、哪一个是
residual。任务名、calibration frequency、概率、entropy 和 table hit 仅用于事后分组，不进入
Judge prompt。

### 2.3 最终正标签

最终只把以下标签计为局部误拒候选：

```text
LOCALLY_SUBSTITUTABLE
```

它又分为：

- `SEMANTIC`：语义、命题、指代、话语功能或推理含义等价；
- `REPRESENTATIONAL`：标点、空白、大小写、分词、数学记号或格式差异不改变任务相关功能。

以下均不计入：

- `BOTH_VALID_DIFFERENT`：两边都局部成立，但改变数量、指代、命题、操作或执行行为；
- `A_UNSAFE/B_UNSAFE/BOTH_UNSAFE`：可见上下文本身已经证明某边无效；
- `INSUFFICIENT_BOUNDARY`：token 本身为空、损坏或不可解析。

V5.5.1 强制 Judge 先解析 token，再比较任务相关作用。只要候选含有可读字符、片段、数字、
运算符、标点或空白，就不能因为句子尚未结束而使用 `INSUFFICIENT_BOUNDARY`。程序还校验
relation、局部状态、equivalence basis、task effect 和 `substitution_safe` 的一致性；非法 JSON
或 schema 冲突会重试，最终失败才跳过。

## 3. 最终误拒候选率

### 3.1 总体与分任务结果

| 任务 | 成功判断事件 | 局部可替换 | 候选率 | 近似 95% CI |
| --- | ---: | ---: | ---: | ---: |
| AIME 2025 | 4,833 | 1,150 | 23.79% | 22.59%–25.00% |
| LiveCodeBench v6 | 10,117 | 1,746 | 17.26% | 16.52%–17.99% |
| OlympiadBench Math EN | 7,101 | 1,438 | 20.25% | 19.32%–21.19% |
| 合计 | 22,051 | 4,334 | **19.65%** | **19.13%–20.18%** |

三个任务都观察到显著数量的局部可替换拒绝，但任务间存在差异：AIME 最高，LCB 最低。
一个合理解释是代码 token 的标识符、操作符、缩进和语法结构更容易产生真实行为差异；数学
推理中则有更多话语衔接、标点和表示形式变化。该解释目前只是机制假设，还需要按
`equivalence_basis` 和 token 类型做任务内分解才能确认。

### 3.2 可替换类型

4,334 个局部可替换事件中：

| 类型 | 数量 | 占全部事件 | 占可替换事件 |
| --- | ---: | ---: | ---: |
| `REPRESENTATIONAL` | 2,780 | 12.61% | 64.14% |
| `SEMANTIC` | 1,554 | 7.05% | 35.86% |

因此当前信号主要来自表示层变体，而不是大范围的语义同义替换。这与高频 pair 中大量出现
标点、换行、数学定界符和自然语言/公式边界相符。它支持“exact token match 会拒绝部分
任务无关的表面变化”，但不支持“任意语义分叉都可以强制接受”。

## 4. Frequency table 与误拒候选的关系

### 4.1 Calibration frequency 分桶

| Calibration frequency | Judge 事件 | 局部可替换 | 候选率 |
| --- | ---: | ---: | ---: |
| 0 | 8,868 | 1,336 | 15.07% |
| 1–2 | 2,902 | 541 | 18.64% |
| 3–5 | 1,633 | 298 | 18.25% |
| 6–9 | 974 | 169 | 17.35% |
| 10–99 | 3,882 | 846 | 21.79% |
| ≥100 | 3,792 | 1,144 | 30.17% |

低频区间不是严格单调：`6–9` 低于 `3–5`。但频次达到 10 后候选率明显升高，尤其
`frequency >= 100` 达到 30.17%。这表明 frequency 不是充分条件，却具有可用的群体富集
作用。

### 4.2 正式活动表阈值 `frequency >= 6`

| 分组 | Judge 事件 | 局部可替换 | 候选率 |
| --- | ---: | ---: | ---: |
| frequency < 6 | 13,403 | 2,175 | 16.23% |
| frequency ≥ 6 | 8,648 | 2,159 | **24.97%** |

活动表相对非活动表：

- 候选率绝对增加 **8.74 个百分点**；
- 候选率相对比值约为 **1.54 倍**；
- 两组差值的朴素 95% 区间约为 **7.63–9.84 个百分点**。

这里的区间仅反映事件抽样误差，不包含 Judge 系统偏差、同一 pair 多次出现造成的组内相关、
任务混杂或 calibration/评测分布差异，因此不能当作严格因果置信区间。

### 4.3 Frequency 到底支撑什么

当前 V5 结果比早期 replay 分析更支持 frequency table，但应精确表述为：

> 在这三个任务的首拒事件中，calibration frequency 较高的 pair 更富集局部可替换事件；
> `frequency >= 6` 的事件级候选率为 24.97%，低频组为 16.23%。

它尚不能单独证明：

1. 同一个高频 pair 在所有上下文中都安全；
2. frequency 越高，安全率必然逐点单调增加；
3. table hit 后应该无条件 force accept；
4. 24.97% 会一比一转化为投机成功率或吞吐提升。

因此 frequency table 适合作为候选检索器，而 probability ratio、entropy gate 或更细的
pair/context 置信度仍承担安全筛选作用。

### 4.4 Calibration table Top 20% pair

Calibration table 共有 633,988 个不同的有向 pair。按 `frequency` 从高到低排序，取前
`ceil(633988 × 20%) = 126,798` 个 pair 为高频 Top 20%，其余表内 pair 为 Bottom 80%。

| Calibration pair 分组 | Judge 事件 | Judge正标签 | 候选率 | 覆盖全部正标签 |
| --- | ---: | ---: | ---: | ---: |
| Top 20% 高频 pair | 11,468 | 2,675 | **23.33%** | 61.72% |
| Bottom 80% 表内 pair | 1,715 | 323 | **18.83%** | 7.45% |
| Calibration 中不存在 | 8,868 | 1,336 | **15.07%** | 30.83% |

如果“低频组”只指 table 内剩余80% pair，则 Top 20% 与 Bottom 80% 的候选率是 23.33% 对
18.83%，相差4.49个百分点。如果把 calibration 中不存在的 pair 也归入“非 Top 20%”，则：

| 二分口径 | Judge 事件 | Judge正标签 | 候选率 |
| --- | ---: | ---: | ---: |
| Calibration Top 20% | 11,468 | 2,675 | **23.33%** |
| 非 Top 20%（Bottom 80% + 未见） | 10,583 | 1,659 | **15.68%** |

二分后绝对差为7.65个百分点，高频组候选率约为低组的1.49倍；Top 20% pair 覆盖52.01%的
Judge事件和61.72%的Judge正标签。

但这个“精确20%”存在并列边界问题：633,988个 pair 中有533,326个（84.12%）的频次等于1，
而所有 `frequency >= 2` 的 pair 只有100,662个（15.88%）。因此Top 20%的最后4.12%只能从
frequency=1的海量并列 pair 中按固定 key 次序任意选取。为避免拆分并列项，更稳健的结果是：

| 不拆并列频次 | Judge 事件 | Judge正标签 | 候选率 |
| --- | ---: | ---: | ---: |
| frequency ≥2（table前15.88%） | 11,345 | 2,669 | **23.53%** |
| frequency =1 | 1,838 | 329 | **17.90%** |
| frequency =0 / calibration未见 | 8,868 | 1,336 | **15.07%** |

这个稳健口径呈 `23.53% > 17.90% > 15.07%`，与Top 20%结果方向一致：calibration 中重复
出现至少两次的 pair 更富集 Judge 正标签。但这些仍是 **Judge candidate rate**，不是人工
校正后的真实可替换率。

## 5. 全体拒绝事件的长尾背景

完整 trace 有 144,863 种有向 pair。分布显著长尾：

| 指标 | 数值 |
| --- | ---: |
| singleton pair 占不同 pair | 69.66% |
| singleton pair 占全部事件 | 21.82% |
| frequency ≥10 的 pair 占不同 pair | 3.86% |
| frequency ≥10 的 pair 覆盖全部事件 | 51.17% |
| Top 1% pair 覆盖全部拒绝 | 36.44% |
| Top 5% pair 覆盖全部拒绝 | 54.18% |
| Top 20% pair 覆盖全部拒绝 | 71.70% |

这说明 frequency table 的第一层价值是压缩：少量 pair 能覆盖大量实际拒绝事件。V5 又提供了
第二层证据：活动频次组的局部可替换率高于非活动组。二者必须同时表达：高覆盖来自长尾，
富集能力来自 V5 direct-local 判断。

### 5.1 使用评测 trace 自身频率复算

原 V5 summary 的 `table_frequency` 来自外部 RedPajama calibration table。补充使用三任务完整
trace 自身的有向 pair 频率复算后得到：

| 三任务合并本地频率 | Judge 事件 | 可替换事件 | Judge 候选率 |
| --- | ---: | ---: | ---: |
| 1 | 4,723 | 572 | 12.11% |
| 2 | 1,790 | 250 | 13.97% |
| 3–5 | 2,611 | 438 | 16.78% |
| 6–9 | 1,709 | 323 | 18.90% |
| 10–99 | 6,002 | 1,138 | 18.96% |
| ≥100 | 5,216 | 1,613 | 30.92% |

任务内部频率也从 singleton 的 13.20% 升至 ≥100 的 34.72%。本地 pooled frequency ≥6
覆盖 3,074/4,334=70.93% 的 Judge 正标签；frequency ≤5 仍有 1,260 个正标签，占 29.07%。
因此本地频率提供了更清晰的富集证据，但低频局部等价候选仍不可忽略。

### 5.2 低频语义候选案例

下面6个 pair 均来自 `pooled_trace_frequency=1` 的独立随机样本。这里的“频率1”只表示该
**有向 token pair** 在三任务合计190条生成轨迹的首拒事件中出现一次，不表示两个英语单词
本身在语料中罕见。

| Draft → Residual | 中文含义 | Calibration / pooled / task频率 | 结合当前上下文的判断 |
| --- | --- | ---: | --- |
| `region` → `subset` | 区域 → 子集 | 0 / 1 / 1 | 有关联但不严格等价；区域是具有几何/连通性质的子集，而任意子集未必是区域，可能损失技术精度 |
| `assigned` → `selected` | 被分配、被指定 → 被选择 | 0 / 1 / 1 | 当前元素角色分配语境中较可能保持含义，但“选中”不总等于“分配到某角色” |
| `mapping` → `transformation` | 映射 → 变换 | 0 / 1 / 1 | 在“把圆映到正方形以改变表示”的语境中较可能安全；一般数学中 mapping 比 transformation 更宽泛 |
| `component` → `part` | 组成部分、连通分量 → 部分 | 1 / 1 / 1 | 当前只泛指所构造的烷烃子图时可能安全；若 `component` 特指连通分量，则 `part` 不够精确 |
| `imply` → `state` | 蕴含、意味着 → 陈述、说明 | 0 / 1 / 1 | 高度可疑；逻辑上“由前提推出”与“文字明确陈述”不是同一关系，不能仅因都能引出命题而判等价 |
| `values` → `vector` | 数值、取值 → 向量 | 1 / 1 / 1 | 高度可疑；向量是带结构的数学对象，不等于一组数值，除非后文明确把该有序列表视为向量 |

这组样本同时说明两点。第一，低频区确实存在可能的合法语义替换，例如当前上下文中的
`mapping/transformation`；frequency threshold 会漏掉这类一次性表达。第二，Judge 会把“相关
概念”“都能开始合理续写”误判为“局部可互换”，例如 `imply/state` 和 `values/vector`。因此
低频正标签数量不能直接解释为低频真实误拒数量。

这些常见单词仍形成低频 pair，主要有四个原因：

1. 统计单位是有方向的 token ID pair；`mapping→transformation` 与反方向是两个 pair；
2. 当前完整 trace 只有190条生成轨迹，却产生144,863种不同 pair，69.66% 的 pair 只出现一次；
3. 同一个词有许多可能的 residual 竞争项，事件会分散到多个具体 pair，而不会聚合成“同义词类”；
4. Calibration 来自 RedPajama，当前评测是数学和代码，领域与上下文分布不同。

所以“低频”与“语义上常见或合理”并不矛盾。Frequency table 学到的是某个 tokenizer 下特定
有向 pair 的经验重复度，不是词典层面的同义词频率。

详细案例审计与本地频率统计见：

```text
DIRECT_LOCAL_LABEL_AND_FREQUENCY_AUDIT_CN.md
RANDOM_LABEL_SAMPLES_20_REVIEW_CN.md
```

## 6. 为什么早期结果不能作为最终误拒率

| 版本 | 样本/方法 | 主要结果 | 应如何解释 |
| --- | --- | --- | --- |
| V1 | 9,645 条，有限续写 | 严格标签占全部事件 69.36%，占可判断事件 90.64% | Prompt 把“未发现错误/两边流畅”过度判为等价，明显偏宽松，只保留作历史来源 |
| V2 | 同一批 replay，严格路径等价 | 严格证据 7.69%；合理但未证明 39.37%；不足 51.19% | 给出保守路径等价下界，但会漏掉不同且正确的推理路径 |
| V3 | 独立判断两边有效性 | draft 可证明有效 5.56%；等价路径下界 3.87%；合理未证明 46.08% | 解除了“必须同一路径”的限制，但有限续写仍难证明最终正确性 |
| V4 | 更严格区分不足与合理未证明，3% 大样本 | draft 可证明有效 5.69%；等价下界 4.17%；合理未证明 39.72%；明确无效 0.88% | 更稳定的 replay 诊断，仍回答路径有效性而非局部替换性 |
| V5.5.1 | 不续写，完整前缀 + 单 token | 局部可替换 19.65% | 当前最贴近 CSD 局部 force-accept 机制的事件级候选率 |

这些数字并不矛盾，因为版本改变了问题：V2–V4 要求有限 continuation 提供路径正确证据；
V5 只要求当前 token 的局部替换不改变已建立的任务相关作用。V1 的 69.36% 则主要来自定义
过宽，不能与 V5 的 19.65%并列当作同一指标的不同估计。

## 7. GSM8K 单 token 反事实实验的补充证据

另一组实验从 GSM8K 全量 trace 中抽取 3% 的拒绝事件，对每个事件分别强制 draft token 和
residual token，然后各自生成到答案。6,836 个事件的结果为：

| 配对结果 | 数量 | 比例 |
| --- | ---: | ---: |
| 两边都正确 | 1,730 | 25.31% |
| 仅 draft 正确 | 751 | 10.99% |
| 仅 residual 正确 | 814 | 11.91% |
| 两边都错误 | 3,541 | 51.80% |

Draft 分支准确率为 36.29%，residual 分支为 37.21%，差异为 -0.92 个百分点；答案变化率为
31.01%。这个实验说明单次替换在总体准确率上没有形成很大的平均差距，也确实存在
`draft-only-correct` 事件。但它不能测量真实 CSD 的累计效果：每个样本只干预一个位置，
后续没有持续 force accept；生成随机性和长轨迹分叉也会掩盖局部等价关系。因此它是安全性
与终局行为的补充诊断，不是最终局部误拒率定义。

## 8. 从事件级误拒率推导成功率提升

设：

- $S$：bare 的 draft 槽位成功率；
- $m$：首拒事件中被判局部可替换的比例，当前估计为 19.65%；
- $Q$：verify round 数；
- $R$：记录到首拒的 round 数；
- $K$：每轮实际参与统计的 draft 槽位数；
- $\bar c$：纠正当前误拒后，平均额外恢复的后续 draft 数；
- $\bar g=1+\bar c$：每个误拒事件平均恢复的总 draft 槽位数。

在固定 round 和固定槽位分母的近似下：

$$
\Delta S=m\frac{R}{Q}\frac{1+\bar c}{K}.
$$

当前三个 trace 的 $R/Q$ 为：

| 任务 | Verify rounds $Q$ | 首拒事件 $R$ | $R/Q$ |
| --- | ---: | ---: | ---: |
| LCB | 381,743 | 212,028 | 55.54% |
| AIME | 231,195 | 101,511 | 43.91% |
| OlympiadBench | 320,362 | 148,908 | 46.48% |

例如仅为了量级说明，取 $m=19.65\%$、$R/Q=50\%$、$K=3$：

| 平均后续级联 $\bar c$ | 每事件总恢复 $\bar g$ | 理论成功率绝对提升 |
| ---: | ---: | ---: |
| 0.0 | 1.0 | 3.28 个百分点 |
| 0.5 | 1.5 | 4.91 个百分点 |
| 1.0 | 2.0 | 6.55 个百分点 |
| 1.5 | 2.5 | 8.19 个百分点 |
| 2.0 | 3.0 | 9.83 个百分点 |

该表是假设所有 Judge 正标签都通过运行时门控且真正恢复对应 token 的理论量级，不是已测
端到端增益。真实结果还会受到 table coverage、probability ratio、entropy gate、误判、拒绝
位置、动态改变后续 verify rounds 以及 kernel 开销影响。

## 9. 当前可以与不可以提出的主张

### 可以提出

1. Exact-match verifier 的首拒事件中存在可观的局部表面或语义等价候选；V5.5.1 Judge
   候选率为 19.65%，其中约 64% 是 representational variant，但其人工校正 precision 尚未测量。
2. 拒绝 pair 呈明显长尾，Top 20% pair 覆盖 71.70% 的拒绝事件。
3. Calibration frequency 对 direct-local 正标签有富集：`frequency >= 6` 为 24.97%，
   `<6` 为 16.23%。
4. Frequency 适合缩小候选空间，但不足以支持无条件接受，仍需要概率和熵门控。

### 暂时不能提出

1. “19.65% 的失败 draft token 都是误拒”；分母不一致。
2. “纠正 19.65% 误拒就必然提升 19.65 个百分点”；需要级联和槽位换算。
3. “高频 pair 在任意上下文中都等价”；当前结论是事件分布上的富集。
4. “Judge 标签是真实 ground truth”；随机抽检已经发现明显假阳性，需要分层人工盲审校正。
5. “局部可替换必然保持完整任务精度”；需要累计 force-accept 的端到端精度实验。

## 10. 下一步最关键的实验

1. 在 trace 中记录首拒位置、剩余 draft 数以及若 force accept 后可继续接受的 token 数，直接
   测量每事件 $g_e$，而不是假设平均级联长度。
2. 报告 token 加权恢复率：

   $$
   \rho=\frac{\sum_e g_e}{KQ(1-S)}.
   $$

3. 按 pair 聚合 `safe occurrences / judged occurrences`，使用请求或 pair cluster bootstrap，
   避免把同一高频 pair 的重复事件当作完全独立样本。
4. 对高频活动表、低频表和 calibration 未见 pair 分层人工盲审，检查 Judge 系统偏差。
5. 将 V5 标签与运行时 `table_hit -> ratio gate -> entropy gate -> forced_accept` 对齐，报告每层
   precision、coverage 和最终 token 恢复量。

## 11. 结果来源

```text
# 最终 Direct Local Judge
runs/mtp_csd/qwen35b_mtp314_rejection_trace/runs/
  direct_local_v5_sample5_20260815_000100/
    sample_manifest.json
    judged_v5_direct_local.jsonl
    judge_errors.jsonl
    summary_v5_direct_local.json
    summary_v5_direct_local.md

# 早期 V1–V3 replay
runs/mtp_csd/qwen35b_mtp314_rejection_trace/runs/
  full_pipeline_20260813_015303/

# V4 replay 与长尾分析
runs/mtp_csd/qwen35b_mtp314_rejection_trace/runs/
  large_v4_sample3_20260814_015007/

# GSM8K 单 token 反事实实验
runs/mtp_csd/qwen35b_mtp314_rejection_trace/runs/
  gsm8k_full_counterfactual_judge_20260814_194026/
```
