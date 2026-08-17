# Direct Local Judge 标签与本地频率审计

## 1. 审计问题

本次补充审计回答三个问题：

1. 随机抽取的 `BOTH_VALID_DIFFERENT` 是否都符合“两个 token 均局部成立，但任务作用不同”；
2. `LOCALLY_SUBSTITUTABLE` 在高频与低频 pair 中分别长什么样，低频正标签是否仍然很多；
3. 原汇总使用的是 calibration table frequency 还是评测 trace 自身 frequency；若是前者，改用
   本地频率后关系如何。

审计输入为最终 V5.5.1 的 22,051 条合法判断。抽样和频率重算脚本为：

```text
analyze_direct_local_trace_frequency.py
```

机器可读结果为：

```text
runs/direct_local_v5_sample5_20260815_000100/trace_frequency_audit.json
```

以上路径均相对于：

```text
runs/mtp_csd/qwen35b_mtp314_rejection_trace/
```

## 2. 先确认：原汇总使用的是 calibration frequency

原始 `summary_v5_direct_local.json` 按下面的字段分桶：

```python
source_event["table_frequency"]
```

该字段来自正式加载的 RedPajama 六领域 calibration table，不是 LCB、AIME 和
OlympiadBench trace 自身的出现次数。因此原来的：

```text
frequency = 0 / 1-2 / 3-5 / 6-9 / 10-99 / >=100
```

全部是 **calibration frequency**。

本次新增两种本地频率：

- `pooled_trace_frequency`：有向 `(draft_token_id, residual_token_id)` pair 在三个完整评测
  trace 合并后的出现次数；
- `task_trace_frequency`：同一个有向 pair 在当前任务内部的出现次数。

Pair 保持方向，不合并 `draft→residual` 和 `residual→draft`。

## 3. `BOTH_VALID_DIFFERENT` 抽查

### 3.1 总体判断

最终结果中有 17,705 个 `BOTH_VALID_DIFFERENT`。初次固定 `seed=42` 抽取了 12 条进行探索，
随后又以 `seed=20260815` 独立抽取 10 条并完整公开。由于尚未制定双人或多人一致的人工标注
协议，本报告不再用“多数合理”概括这些样本，也不据此给出 precision 数字。能够直接观察到：

- 样本中同时存在直观不同的 pair 和边界合法性可疑的 pair；
- 但“两个候选均 VALID”的论证并不总是成立；
- 个别 evidence 会想象候选后的续写、误读 token 拼接位置，甚至把 A/B 的作用说反；
- 因此不能认为这一标签集合已经全部符合严格定义。

独立随机样本的原始上下文、候选、三种频率和 Judge evidence 均保存在：

```text
runs/direct_local_v5_sample5_20260815_000100/RANDOM_LABEL_SAMPLES_CN.md
```

### 3.2 较可信的不同样本

| Draft → Residual | 上下文作用 | 审计判断 |
| --- | --- | --- |
| `.` → `?` | 陈述结束与疑问结束不同 | 关系标签合理，但是否都能自然接在该边界仍需看精确拼接 |
| `If` → `For` | 条件结构与循环/枚举结构不同 | 不可替换判断合理 |
| `If` → `Here` | 条件连接与指示连接不同 | 不可替换判断合理 |
| `A` → `C` | 指向不同几何点 | 不可替换判断合理 |
| `=` → `$` | 等式操作符与数学环境定界符不同 | 不可替换判断合理 |
| `be` → `count` | 状态/被动结构与计数概念不同 | 不可替换判断基本合理 |

### 3.3 存在问题的不同样本

| Draft → Residual | 问题 |
| --- | --- |
| `that` → `the` | 可见前缀已经以 `could the` 结束；Judge 没有严格检查 token 拼接后的语法，却声称两边都有效 |
| `"` → `block` | Evidence 对 A/B 的作用描述与实际 token 对应混乱，并想象不存在的代码块结构 |
| `are` → `also` | Judge 用“可能继续为……”解释两边，依赖未显示 continuation，违反只看当前边界的原则 |
| `The` → `Lines` | Evidence 推断 `Lines` 是章节标题，但可见上下文不足以确定这一具体结构 |

因此，`BOTH_VALID_DIFFERENT` 可以较可靠地作为“不应 force accept”的保守集合，但不能把它
解释为经过人工验证的双边有效集合。后续人工审计最好拆成两个问题：

```text
Q1: 两边是否各自局部合法？
Q2: 若都合法，替换是否改变任务相关作用？
```

## 4. 可替换标签抽查

### 4.1 低频正标签并不少

以三任务合并本地频率计算：

| 条件 | `LOCALLY_SUBSTITUTABLE` 事件 | 占全部正标签 4,334 的比例 |
| --- | ---: | ---: |
| pooled frequency = 1 | 572 | 13.20% |
| pooled frequency ≤5 | 1,260 | 29.07% |
| pooled frequency ≥6 | 3,074 | 70.93% |
| pooled frequency ≥100 | 1,613 | 37.22% |

若按任务内部频率，frequency=1 的正标签有 792 个，占 18.27%。所以低频等价事件仍然很多，
frequency table 必然牺牲一部分召回；它的目标是优先覆盖更常见、总体贡献更大的 pair，而不是
穷尽所有局部等价表达。

### 4.2 低频正标签中的合理案例

随机样本中存在直观上可能安全的 pair：

| Draft → Residual | 本地合并频率 | 审计意见 |
| --- | ---: | --- |
| `down` → ` down` | 1 | 仅前导空格不同，若不处于代码/字面量中，可能是表示层等价 |
| `potentially` → `maybe` | 1 | 在表达不确定性的普通推理语境中可能语义等价 |
| `determined` → `fully` | 1 | 具体样本实际依赖后续词序，只能算可疑正例，不能仅凭词义认定 |

### 4.3 低频正标签中的明显或高度可疑假阳性

| Draft → Residual | 本地合并频率 | 为什么不应直接判安全 |
| --- | ---: | --- |
| 14 个空格 → 13 个空格 | 1 | Python 缩进可能改变代码块和执行行为 |
| `circle` → `Disk` | 1 | circle 常指边界，disk 指内部区域，数学对象不同 |
| `left` → `prev` | 1 | 只替换一次变量名而不一致重命名，可能改变程序行为 |
| `number` → `carry` | 1 | 指代不同概念，不是同义替换 |
| `notation` → `logic` | 1 | 数学表示与推理逻辑不同 |
| `special` → `property` | 1 | 词性和预期后续结构不同 |
| `becomes` → `was` | 4 | 变化过程与过去状态不同 |
| `standard` → `strict` | 3 | 含义不同，可能改变要求 |

这些反例说明 V5.5.1 虽然解决了 `INSUFFICIENT_BOUNDARY` 滥用，却仍会把“都能开启某种合理
续写”误判为“当前可互换”。

### 4.4 高频正标签中的合理案例

| Draft → Residual | 本地合并频率 | 审计意见 |
| --- | ---: | --- |
| `).` → `)` | 258 | 若只差句末可选标点，通常是表示层等价 |
| `}` → `}$` | 607 | 可能只是 tokenizer 对 LaTeX 结束边界的不同切分，但必须核对上下文中 `$` 是否已配平 |
| `the` → `that` | 157 | 在指代同一已知概率等少数上下文中可能等价，不可只按 pair 无条件接受 |
| `Let` → `We` | 118 | 可能承担相同推理步骤引导功能，但通常还依赖后续句法 |

### 4.5 高频正标签中仍有明显假阳性

| Draft → Residual | 本地合并频率 | 问题 |
| --- | ---: | --- |
| `$.'` → `$,` | 589 | 句号结束陈述、逗号继续句子，不能在未知后续时自动视为等价 |
| 空格 → `$` | 1,281 | 空白与数学模式定界符的结构作用不同 |
| `\` → `<` | 172 | LaTeX 命令前缀与比较运算符明显不同 |
| 换行 → `So` | 618 | 结构分隔符与话语连接词不是同一 token 作用 |
| 空格 → `\` | 662 | 在 `\sqrt` 等命令附近会改变 LaTeX 结构 |
| `$.'` → `\` | 662 | 数学环境结束与命令前缀不同 |

高频并没有消除 Judge 假阳性。频率可以提高总体正标签比例，但不能替代上下文安全门或更可靠
的人工/模型判定。

## 5. 使用本地自身频率重新统计

### 5.1 三任务合并频率

| Pooled trace frequency | Judge 事件 | 可替换事件 | 可替换率 |
| --- | ---: | ---: | ---: |
| 1 | 4,723 | 572 | 12.11% |
| 2 | 1,790 | 250 | 13.97% |
| 3–5 | 2,611 | 438 | 16.78% |
| 6–9 | 1,709 | 323 | 18.90% |
| 10–99 | 6,002 | 1,138 | 18.96% |
| ≥100 | 5,216 | 1,613 | 30.92% |

这个趋势比 calibration frequency 分桶更清晰：从 singleton 的 12.11% 逐步升至高频头部的
30.92%，仅 `6–9` 与 `10–99` 基本持平。

以本地阈值 6 分组：

| 分组 | Judge 事件 | 可替换事件 | 可替换率 | 对全部正标签的覆盖 |
| --- | ---: | ---: | ---: | ---: |
| pooled frequency <6 | 9,124 | 1,260 | 13.81% | 29.07% |
| pooled frequency ≥6 | 12,927 | 3,074 | 23.78% | 70.93% |

因此本地高频组同时具备：

- 更高的 Judge 正标签率；
- 只保留 58.62% 的抽样事件，却覆盖 70.93% 的正标签。

### 5.2 任务内部频率

| Task-local frequency | Judge 事件 | 可替换事件 | 可替换率 |
| --- | ---: | ---: | ---: |
| 1 | 6,001 | 792 | 13.20% |
| 2 | 2,204 | 360 | 16.33% |
| 3–5 | 2,943 | 512 | 17.40% |
| 6–9 | 1,808 | 340 | 18.81% |
| 10–99 | 5,907 | 1,223 | 20.70% |
| ≥100 | 3,188 | 1,107 | 34.72% |

任务内部频率也呈相同趋势，且头部 ≥100 达到 34.72%。这说明本地频率与 Judge 正标签的关系
不是单纯由三个任务合并后共享标点造成的。

### 5.3 与 calibration frequency 对比

| 频率来源 | 低端正标签率 | 高频头部正标签率 | 特点 |
| --- | ---: | ---: | --- |
| Calibration table | frequency=0：15.07% | ≥100：30.17% | 可在正式运行前获得，适合在线查表 |
| 三任务合并 trace | frequency=1：12.11% | ≥100：30.92% | 使用评测集事后统计，趋势更平滑 |
| 当前任务内部 trace | frequency=1：13.20% | ≥100：34.72% | 最贴近任务分布，但在线冷启动时不可提前知道 |

本地频率结果更强地支持“重复出现的 pair 更可能属于局部可替换候选”。但本地频率是评测后
才能得到的 oracle-like 特征，不能直接替代正式 calibration table；它更适合证明机制、设计
task-matched calibration，或评估 dynamic update 的潜在上限。

## 6. 修订后的结论

### 6.1 可以保留的结论

1. 无论使用外部 calibration frequency 还是本地 trace frequency，高频头部都更富集
   `LOCALLY_SUBSTITUTABLE` 标签。
2. 本地频率趋势尤其明显：pooled singleton 为 12.11%，pooled ≥100 为 30.92%；任务内
   singleton 为 13.20%，任务内 ≥100 为 34.72%。
3. 本地 pooled frequency ≥6 覆盖 70.93% 的正标签，说明 frequency table 具有候选压缩和
   召回价值。
4. 低频正标签仍占约 29%，所以 frequency threshold 存在明确的召回损失。

### 6.2 必须弱化的结论

`LOCALLY_SUBSTITUTABLE=19.65%` 不能称为已经验证的严格误拒率。随机案例中出现多种明显
假阳性，说明 Judge precision 尚未测定。当前更严谨的名称应是：

```text
Direct-local Judge candidate rate = 19.65%
```

Frequency 分析证明的是“高频对 Judge 正标签有富集”，而不是“高频 pair 已被证明可以安全
force accept”。要得到严格误拒率，需要对分层样本做人工盲审，估计每组 Judge precision：

$$
\widehat{m}_{\mathrm{corrected}}
=\sum_b P(b)\,P(\text{Judge positive}\mid b)\,
  \widehat{P}(\text{truly substitutable}\mid\text{Judge positive},b).
$$

建议分层至少包括任务、pooled frequency、calibration hit、semantic/representational basis 和
token 类型。
