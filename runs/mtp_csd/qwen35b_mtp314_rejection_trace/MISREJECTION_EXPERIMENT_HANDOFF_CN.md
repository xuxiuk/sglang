# CSD 误拒识别实验 Handoff

## 1. Handoff 目的

本文档用于让后续接手者不依赖聊天记录即可理解：我们想测什么、为什么多次改变实验定义、
代码如何实现、各版本结果说明什么、哪些结论已经成立，以及下一步怎样把事件级 Judge 信号
连接到真实投机成功率与吞吐收益。

当前最终判断口径是 **V5.5.1 Direct Local Substitution**。V1–V4 和 GSM8K counterfactual
仍保留，因为它们分别揭示 prompt 偏差、有限续写的证据不足，以及单 token 干预对最终答案
的影响；但它们不应混合成一个“误拒率”。

## 2. 最初要解决的问题

标准 speculative decoding 使用 target verifier 检查 draft token。只要 draft token 与 target
在当前位置选择的 residual token 不一致，就发生拒绝。然而 token 不同并不总意味着任务相关
作用不同。例如：

- 句号与逗号可能只改变非关键句子分段；
- 单换行与双换行可能只改变格式；
- 两种数学定界方式可能表示同一结构；
- “therefore”与“thus”可能表达相同推理关系；
- 不同 tokenization 可能还原为同一局部文本或功能。

CSD 的核心假设是：某些重复出现的 `(draft_token, residual_token)` pair 在许多上下文中属于
exact-match verifier 的不必要拒绝。Frequency table 先定位反复出现的 pair，再由 probability
ratio 和 entropy gate 限制 force accept。

实验要回答五个层次不同的问题：

1. 拒绝 pair 是否呈长尾，少量 pair 能否覆盖大量拒绝？
2. 一次拒绝是否属于局部不必要拒绝？
3. 高频 pair 是否更富集这种局部误拒？
4. 单次替换是否改变最终答案正确性？
5. 纠正一个首拒后能恢复多少 draft token，最终提高多少投机成功率和吞吐？

其中 1–4 已有实验；第 5 项目前只有公式和理论量级，尚缺实际级联 trace。

## 3. 关键术语与分母

### 3.1 首拒事件

当前 trace 每个 verify round 记录第一个被 verifier 拒绝的位置。一次 round 最多产生一个
首拒事件。因此 Judge 的基本样本单位是“首拒事件”，不是所有 draft 槽位。

### 3.2 失败 draft 槽位

失败 draft 槽位是所有被提出但最终没有被 verifier 接受的 draft token 位置。若一轮有 $K$
个参与统计的 draft 槽位，执行 $Q$ 轮，总接受 draft 数为 $A$，则：

$$
S=\frac{A}{KQ},\qquad N_{\mathrm{failed}}=KQ-A.
$$

同一个首拒事件之后可能还剩多个未接受槽位，所以事件级误拒率不等于失败槽位级误拒率。

### 3.3 当前最终指标

$$
m=\frac{\text{LOCALLY_SUBSTITUTABLE 首拒事件}}
        {\text{成功判断的首拒事件}}.
$$

当前 $m=4334/22051=19.65\%$。

若要转成成功率提升，需要：

$$
\Delta S=m\frac{R}{Q}\frac{\bar g}{K},
\qquad \bar g=1+\bar c.
$$

$R/Q$ 是发生首拒的 round 比例，$\bar c$ 是纠正当前误拒后平均额外恢复的后续 draft 数。

## 4. 数据采集实现

### 4.1 Trace 设计

基于 `/root/sglang-dspark-csd` 的 MTP Python 验证路径实现 rejection trace。正常 verifier
执行结束后，Python 层已有：

```text
candidates
target_probs
target_logits
predict
accept_index
num_correct_drafts
```

因此不需要改变正常 force-accept 结果即可恢复首拒位置、draft token、residual token、target
概率和 entropy。Trace 模式关闭 CSD force accept，只观察 bare verifier；额外参数控制是否
落盘，避免默认推理承担记录开销。

每个请求级记录保存一次完整 prompt token IDs 和生成 token IDs；拒绝事件单独保存请求标识、
生成位置、draft/residual token、概率、entropy、table frequency/hit 等。后处理通过请求标识和
`generated_position` 恢复：

```python
context_ids = prompt_token_ids + generated_token_ids[:generated_position]
```

这种设计避免每个事件重复写完整上下文，同时能精确恢复拒绝边界。

### 4.2 正式 trace

```text
runs/mtp_csd/qwen35b_mtp314_rejection_trace/runs/
  trace_large_80_30_80_20260814_001505/
```

任务规模：LCB 80、AIME 30、OlympiadBench Math EN 80，每题一条轨迹。三任务原始首拒事件
合计 462,447，去重后 462,441。

### 4.3 写入与容错

Trace 采用运行结束 flush/barrier，确保请求记录和事件记录完整落盘。Judge 阶段采用：

- HTTP 并发 64；
- 单请求超时 7200 秒；
- 最多重试 3 次；
- temperature 0；
- JSON schema 与跨字段一致性检查；
- 永久失败写入独立 error JSONL，跳过后继续；
- `RESUME=1` 时跳过已经成功的 sample，重新尝试失败项。

因此单个 HTTP、解析或后处理错误不会终止全量实验。

## 5. 实验路线为何演进

### 5.1 第一阶段：续写两个分支

最初方案在拒绝位置分别选择 draft 和 residual，然后各自续写固定 token 数，再让大模型判断
两条 continuation 是否正确或等价。优点是能看到分叉后的行为；缺点是：

1. 一个 token 会诱导不同但都可能正确的推理路径；
2. 有限 128/256 token 往往不足以到达答案；
3. “未完成”容易被 Judge 误判为 insufficient；
4. continuation 的采样噪声掩盖了拒绝位置本身的局部关系；
5. CSD 的 frequency table 本来描述局部 pair，而 replay 回答的是长程路径问题。

这导致 V1–V4 的 prompt 不断在“太宽松”和“太严格”之间调整。

### 5.2 第二阶段：GSM8K 生成到最终答案

为了避免有限续写没有答案，GSM8K 实验对随机 3% 的全部拒绝事件分别强制一个 draft 或
residual token，并继续生成到答案。它能测单次 token 干预的终局影响，但仍只干预一个位置，
不等价于真实 CSD 在整条轨迹上多次 force accept。

### 5.3 第三阶段：Direct Local Judge

最终回到 CSD 的原始局部假设：不续写，只给完整共享上下文与两个单 token，判断当前边界是否
局部可替换。它避免把“最终能否答对”混入“当前 exact-match rejection 是否必要”。这是当前
主分析采用的 V5 路线。

## 6. Judge 各版本与结论

### 6.1 V1：Legacy lenient replay Judge

设计：两个分支续写后，判断 `BOTH_VALID_EQUIVALENT`、`BOTH_VALID_DIFFERENT`、
`DRAFT_BETTER` 等。

结果：9,645 个事件中，6,690 个被计为严格误拒；占全部事件 69.36%，占可判断事件 90.64%。

问题：没有观察到错误就容易被判为有效/等价，局部流畅被误当作路径正确。原始 prompt 文本
还曾被覆盖，不能完整复现。

结论：只作为探索历史；69.36% 不能作为误拒率报告。

### 6.2 V2：Strict path-equivalence

设计：只有 continuation 明确证明两边保持相同正确结论或 draft 更好，才算严格证据；未完成
但合理的分支单列 `BOTH_PLAUSIBLE_UNPROVEN`。

结果：

| 指标 | 数量 | 占全部事件 |
| --- | ---: | ---: |
| `PROVEN_EQUIVALENT + DRAFT_BETTER` | 742 | 7.69% |
| `BOTH_PLAUSIBLE_UNPROVEN` | 3,797 | 39.37% |
| `INSUFFICIENT_CONTEXT` | 4,937 | 51.19% |

问题：要求两条分支路径等价过重。代数法与几何法都正确但不等价时，也可能是不必要拒绝，
却不会进入严格正标签。

结论：7.69% 是保守路径等价证据，不是局部误拒率。

### 6.3 V3：Independent validity replay Judge

设计：分别判断 draft 和 residual 是否 `PROVEN_VALID / PROVEN_INVALID /
PLAUSIBLE_UNPROVEN / INSUFFICIENT`，再判断结果和路径关系。不同正确路径不再要求等价。

结果：

| 指标 | 比例 |
| --- | ---: |
| Draft 可证明有效 | 5.56% |
| Draft 强于 residual 的强证据 | 0.35% |
| 两边都可证明有效 | 3.88% |
| 等价路径下界 | 3.87% |
| Draft 合理但未证明 | 46.08% |

问题：有限 continuation 仍很难证明完整答案正确，大量事件停留在 plausible 或 insufficient。

结论：独立有效性定义更合理，但问题仍与 CSD 的局部替换假设不完全一致。

### 6.4 V4：Boundary-constrained independent validity

设计：进一步规定“未完成但已有任务相关步骤”应判 plausible，而不是 insufficient；只有可见的
具体错误才能判 invalid。正式大样本为三任务全局 3%，replay 256 token。

结果：13,873 个事件中：

| Draft 状态 | 数量 | 比例 |
| --- | ---: | ---: |
| `PROVEN_VALID` | 789 | 5.69% |
| `PLAUSIBLE_UNPROVEN` | 5,510 | 39.72% |
| `INSUFFICIENT` | 7,452 | 53.72% |
| `PROVEN_INVALID` | 122 | 0.88% |

早期分析把 `PROVEN_VALID + PLAUSIBLE_UNPROVEN = 45.40%` 称为“潜在误拒候选”。它只能说明
draft 没有显示错误且存在一定正面信号，不能作为 ground-truth 误拒率。

V4 的频率分析发现：固定完整 trace 的 Top 20% pair 覆盖 71.15% 的候选，与覆盖全部拒绝的
71.70% 几乎相同；各频率排名区间候选率约 45%，没有明显单调关系。

结论：V4 说明全体拒绝长尾足以带来高覆盖，但当时的宽候选标签没有显示 frequency 的额外
判别力。后来 V5 使用更匹配问题的局部标签后，观察到了活动频次组的富集。

### 6.5 GSM8K single-intervention counterfactual

设计：从所有拒绝事件全局随机抽取 3%，每次只在一个位置选择 draft 或 residual，分别生成到
答案；不是每请求只选一个事件，也不是累计 CSD。

结果：6,836 个配对事件中，两边都正确 25.31%，仅 draft 正确 10.99%，仅 residual 正确
11.91%，两边都错误 51.80%。Draft 与 residual 总准确率分别为 36.29% 和 37.21%。

结论：单 token 选择在总体终局精度上差异不大，同时两种方向均存在独占正确样本；但随机
长轨迹会放大方差，它不能直接回答局部等价，也不能模拟多次 force accept。

### 6.6 V5.1–V5.4：Direct-local prompt 调整

V5 开始取消 replay，只判断完整前缀后的单 token。中间版本主要修复以下漂移：

1. `LOCAL_EQUIVALENT` 与 `FUNCTIONAL_VARIANT` 重叠，后合并为一个
   `LOCALLY_SUBSTITUTABLE`，用 `equivalence_basis` 区分语义和表示层原因；
2. Judge 因句子、公式或代码未完成而滥用 `INSUFFICIENT_BOUNDARY`；
3. Judge 把“两个 token 都可以开启一条合理续写”误判为可替换；
4. 标点和换行被一律判不同，或反过来被一律判等价；
5. 数字、方向、变量和运算符的真实任务作用没有被明确要求；
6. 输出字段彼此冲突，缺少程序端一致性验证。

这些版本的部分输出和配置保留在最终运行目录的 `.partial` 文件中，只用于 prompt 调试，不能
与完整 V5.5.1 结果拼接统计。

### 6.7 V5.5.1：Resolve then compare

最终规则分两步：

1. 先解析候选 token 本身。只要有任何稳定的可见贡献，就禁止 insufficient；
2. 再要求 Judge 指出替换造成的具体任务相关 effect。没有具体 effect 才能判局部可替换。

最终合法标签只有：

```text
LOCALLY_SUBSTITUTABLE
BOTH_VALID_DIFFERENT
A_UNSAFE
B_UNSAFE
BOTH_UNSAFE
INSUFFICIENT_BOUNDARY
```

结果：22,051 个成功事件中 4,334 个局部可替换，事件级候选率 19.65%。其中
representational 2,780 个，semantic 1,554 个。

随机案例复核随后发现明显假阳性，包括不同代码缩进、数学对象、LaTeX 操作符和非同义概念。
因此它是当前最贴合 CSD 局部问题的 **Judge 候选率**，但不是已经人工验证的误拒率。

## 7. Frequency 与长尾结果

### 7.1 完整 trace 长尾

完整 trace 有 144,863 种有向 pair，69.66% 只出现一次；Top 1%、5%、20% pair 分别覆盖
36.44%、54.18%、71.70% 的全部拒绝。这证明 frequency table 能用较小 pair 集合覆盖大量
事件。

### 7.2 V4 与 V5 为什么看起来不同

V4 使用“draft proven/plausible”作为候选，衡量的是续写路径正面证据。该候选在高低频区间
约为 45%，没有频率单调性。V5 衡量当前 token 的局部可替换性，更贴近 pair frequency 的
含义，得到：

| Calibration frequency | V5 局部可替换率 |
| --- | ---: |
| 0 | 15.07% |
| 1–2 | 18.64% |
| 3–5 | 18.25% |
| 6–9 | 17.35% |
| 10–99 | 21.79% |
| ≥100 | 30.17% |

正式活动阈值 `frequency >= 6` 的候选率为 24.97%，低频组为 16.23%，差 8.74 个百分点，
相对约 1.54 倍。

这不意味着 frequency 单调决定安全性；它说明 calibration 中反复出现的 pair 在评测拒绝事件
中更富集局部可替换现象。Frequency 是候选召回信号，不是最终安全判定。

按 calibration table 的633,988个不同 pair 排名，Top 20% pair 的 Judge 候选率为23.33%，
Bottom 80%表内 pair为18.83%，calibration未见 pair为15.07%。若把未见 pair并入非Top 20%，
则高低组为23.33%对15.68%。但20%边界落在frequency=1的大规模并列项中；不拆并列时，
frequency≥2、frequency=1和未见三组分别为23.53%、17.90%和15.07%，这是更稳健的口径。

补充使用评测 trace 自身频率后，pooled frequency=1/2/3–5/6–9/10–99/≥100 的 Judge
候选率依次为 12.11%、13.97%、16.78%、18.90%、18.96% 和 30.92%；任务内部频率的
singleton 与 ≥100 分别为 13.20% 和 34.72%。本地 pooled frequency ≥6 覆盖 70.93% 的
Judge 正标签，但 frequency ≤5 仍占 29.07%。完整审计见：

```text
DIRECT_LOCAL_LABEL_AND_FREQUENCY_AUDIT_CN.md
```

低频正标签中出现了 `region→subset`（区域→子集）、`assigned→selected`（被分配→被选择）、
`mapping→transformation`（映射→变换）、`component→part`（组成部分→部分）、
`imply→state`（蕴含→陈述）和 `values→vector`（数值→向量）。前几组在特定上下文中可能
保持任务含义，后两组则暴露了把“相关概念”误判为“可替换”的 Judge 假阳性。它们低频并不
代表单词罕见：统计单位是 tokenizer 下的特定有向 pair，三任务只有190条轨迹，而且同一词的
竞争 residual 会把观察分散到不同 pair。

后续又用独立种子对四类各抽20条并逐条初审，结果保存于：

```text
RANDOM_LABEL_SAMPLES_20_REVIEW_CN.md
runs/direct_local_v5_sample5_20260815_000100/RANDOM_LABEL_SAMPLES_20_CN.md
```

这80条显示 `BOTH_VALID_DIFFERENT` 相对稳健，但“双方均合法”仍会失败；三组
`LOCALLY_SUBSTITUTABLE` 样本都存在较多假阳性。尤其高频正标签被标点、空白和 LaTeX
边界主导，当前 Judge 会过度把它们归一化。因此全量频率分桶只能证明高频提高 Judge 正标签率，
尚不能证明高频提高人工确认后的真实可替换率。

## 8. 结果与投机成功率的关系

### 8.1 为什么 19.65% 不能直接加到成功率

19.65% 的分母是首拒事件。失败 draft 槽位则包括每轮所有未接受位置。一个首拒事件可能只
恢复当前 token，也可能因级联继续恢复后续多个 token。

设 $m=19.65\%$、首拒 round 比例 $r=R/Q$、每轮槽位 $K$、平均后续级联 $\bar c$，则：

$$
\Delta S=m r\frac{1+\bar c}{K}.
$$

使用示例 $r=50\%, K=3$：

| 平均后续级联 | 每误拒事件平均恢复 | 成功率绝对提升量级 |
| ---: | ---: | ---: |
| 0 | 1.0 | 3.28 pp |
| 0.5 | 1.5 | 4.91 pp |
| 1 | 2.0 | 6.55 pp |
| 1.5 | 2.5 | 8.19 pp |
| 2 | 3.0 | 9.83 pp |

真实 CSD 只会处理 table hit 且通过 ratio/entropy gate 的子集，所以该表仍是理想量级上限，
不是端到端预测。

### 8.2 下一版 trace 必须增加什么

为了从 Judge 结果得到实测 token 收益，每个首拒事件还应记录：

```text
draft_position_in_round
remaining_draft_slots
normal_accept_prefix_before_rejection
forced_token_accepted
downstream_normal_accept_count
downstream_forced_accept_count
total_recovered_draft_tokens g_e
verify_round_id before/after intervention
```

最终直接统计：

$$
\Delta S=\frac{\sum_e g_e}{KQ},
\qquad
\rho=\frac{\sum_e g_e}{KQ(1-S)}.
$$

## 9. 代码、脚本与结果索引

### 9.1 核心代码

| 文件 | 作用 |
| --- | --- |
| `build_direct_local_samples.py` | 从三任务完整 trace 全局去重、无放回抽样并恢复完整上下文 |
| `judge_direct_local.py` | V5.5.1 盲化 Judge、并发请求、schema 校验、重试和断点续跑 |
| `summarize_direct_local.py` | 汇总总体、任务、calibration frequency 与活动表分组 |
| `judge_prompt.md` | 保存 V1–V4 replay prompt 的版本注册表 |
| `replay_rejection_branches.py` | 旧版双分支 continuation replay |
| `judge_replayed_branches*.py` | V1–V4 replay Judge |
| `replay_gsm8k_counterfactual.py` | GSM8K 单位置 draft/residual 生成到答案 |
| `analyze_misrejection_frequency.py` | V4 calibration frequency 与候选覆盖分析 |
| `rejection_frequency_confidence_analysis.ipynb` | 可重复查看频率、置信度、ratio、entropy 分布 |

以上路径均相对于：

```text
runs/mtp_csd/qwen35b_mtp314_rejection_trace/
```

### 9.2 一键入口

```bash
bash runs/mtp_csd/qwen35b_mtp314_rejection_trace/run_direct_local_judge_5pct.sh
```

默认行为：构造 5% 样本，使用 GPU 0–7 启动 DeepSeek-V4-Flash-DSpark Judge，顺序完成
Judge 和汇总。正式配置为 temperature 0、并发 64、最多 3 次重试。

### 9.3 最终结果

```text
runs/mtp_csd/qwen35b_mtp314_rejection_trace/runs/
  direct_local_v5_sample5_20260815_000100/
```

关键文件：

| 文件 | 内容 |
| --- | --- |
| `config.env` | 正式运行配置和 prompt version |
| `sample_manifest.json` | 输入规模、抽样方法、任务构成 |
| `direct_local_samples.jsonl` | 23,122 个完整上下文样本 |
| `judged_v5_direct_local.jsonl` | 22,051 个合法判断 |
| `judge_errors.jsonl` | 1,071 个最终失败事件 |
| `summary_v5_direct_local.json` | 机器可读正式汇总 |
| `summary_v5_direct_local.md` | 简表 |

## 10. 已知问题与风险

1. **Judge 不是人工 ground truth。** Prompt 多次调整表明标签对定义敏感，必须进行分层人工
   盲审或使用独立 Judge 复核。
2. **失败样本可能带来偏差。** 4.63% 请求最终失败；需要比较失败项与成功项的任务、长度、
   token 类型和 frequency 分布。
3. **事件不完全独立。** 同一请求和同一 pair 会出现多次，普通二项置信区间偏乐观，应做
   request/pair cluster bootstrap。
4. **V5 只测局部替换。** 它不证明一条轨迹上累计多次 force accept 后精度不变。
5. **A/B 对称标签仍需核查。** 某些 token 替换可能具有方向性；CSD pair 是有向的，后续应按
   draft→residual 方向聚合，而不是把反向 pair 合并。
6. **Calibration 域偏移。** Table 来自 RedPajama 六领域，而评测是数学与代码，frequency=0
   事件仍占 8,868/22,051；需要任务匹配 calibration 做消融。
7. **CLI 形状不等于公式中的槽位数。** 成功率换算中的 $K$ 必须按运行指标实际统计口径确认，
   不能仅凭 `num_draft_tokens=4` 或 `steps=3` 的名称推断。

## 11. 推荐的下一步执行顺序

1. 修订 `DIRECT_LOCAL_JUDGE_EXPERIMENT_CN.md`，将过时的 V5.3 prompt 描述更新为
   V5.5.1，或明确标记其为设计草稿，避免与正式代码不一致。
2. 对最终 22,051 条结果按任务 × frequency × equivalence basis 分层抽取 300–500 条人工
   盲审，得到 Judge precision 与混淆矩阵。
3. 分析 1,071 个失败样本是否随机；若不随机，修复后只补跑失败项并重新汇总。
4. 新增级联 trace，测量每个正事件的实际 $g_e$，把事件级 19.65% 转成 token 级恢复率。
5. 将 V5 正标签与 table hit、probability ratio、entropy gate 的每层通过情况关联，找出召回和
   精度损失分别发生在哪一层。
6. 在固定模型、任务、生成参数和并发下运行 bare/plain/dynamic/entropy，联合报告精度、平均
   接受长度、投机成功率、输出吞吐及运行时 counters。

## 12. 当前最终结论

当前实验已经支持一个比“高频 pair 天然等价”更谨慎、也更可靠的故事：

> Exact-match speculative verifier 的首拒事件中，约 19.65% 在完整可见上下文下被 Judge 判为
> 局部可替换，其中多数是表示层差异。拒绝 pair 本身呈明显长尾，且 calibration 活动频次组
> 对局部可替换事件具有约 1.54 倍的富集。因此 frequency table 能有效缩小候选空间，但必须
> 与概率、熵或上下文安全门结合；事件级候选率还需通过真实级联长度转换，才能解释投机成功率
> 与吞吐提升。

这条结论与现有数据一致，同时没有把 Judge 候选率夸大为 ground-truth token 误拒率或端到端
加速比。
