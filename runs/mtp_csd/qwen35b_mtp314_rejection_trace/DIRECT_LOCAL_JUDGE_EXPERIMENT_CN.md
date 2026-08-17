# CSD 拒绝 Pair 的 Direct Local Judge 实验

## 1. 研究问题

本实验直接检验 CSD frequency table 的局部假设：同一个
`(draft_token, residual_token)` pair 反复出现时，两个 token 是否更可能在拒绝位置保持
相同的局部语义或功能。

实验不再生成 counterfactual continuation。每个分支严格截止在发生投机拒绝的候选 token：

```text
Branch A = 完整共享上下文 + Candidate A
Branch B = 完整共享上下文 + Candidate B
```

Judge 只判断该位置的局部替换关系，不预测后续推理路径或最终答案。

## 2. Trace 与抽样

输入为以下三个任务已经完成的完整 rejection trace：

```text
runs/mtp_csd/qwen35b_mtp314_rejection_trace/runs/
  trace_large_80_30_80_20260814_001505/
    lcb_v6/trace/
    aime25/trace/
    olympiad_math_en/trace/
```

三任务合计记录 462,447 个拒绝事件和 144,863 个有向 token pair。正式实验对所有可恢复
事件执行全局无放回均匀抽样，固定 `sample_ratio=0.05`、`sample_seed=42`，预计获得约
23,122 个事件。抽样不读取 calibration frequency、table hit、probability ratio、entropy
或 Judge 标签，因此不会预先偏向高频、低频、命中或未命中事件。

任务名、频次和运行特征只作为落盘元数据，用于事后分组，不发送给 Judge。

## 3. 完整上下文

每条样本使用与旧 replay Judge 相同的完整共享上下文：

```python
context_ids = request["prompt_token_ids"] + generated_token_ids[:generated_position]
context_text = tokenizer.decode(context_ids)
```

因此 `context_text` 已经包含原始任务 prompt 和拒绝位置之前的全部生成内容，不需要在
Judge 请求中额外发送任务名。实验不截断为局部窗口，也不提供候选 token 之后的文本。

## 4. Prompt：V5.3 Direct Local Substitution

### System prompt

```text
You are a rigorous evaluator of a local token substitution at a
speculative-decoding rejection boundary.

You receive the complete shared context immediately before a one-token
divergence and two candidate tokens A and B. Their draft/residual identities
are hidden. Each branch consists exactly of the shared context followed by its
candidate token, and ends immediately after that token. No continuation is
available.

Judge only the local semantic and functional relationship between A and B at
this exact position. Do not predict, imagine, or generate any continuation. Do
not decide whether either branch will eventually reach the correct final
answer.

The central question is whether A and B are locally substitutable at this exact
rejection boundary. Local substitutability means that replacing one candidate
with the other preserves the task-relevant meaning, proposition, referent,
reasoning operation, or executable behavior already established by the visible
context. Surface form need not be identical, but both candidates being
plausible is not enough.

Evaluate grammatical and syntactic role; mathematical operators, quantities,
signs, variables, relations, and logical claims; code identifiers, control
flow, APIs, and executable behavior; discourse function; notation, formatting,
punctuation, and tokenization variants.

Use exactly one relation label:

1. LOCALLY_SUBSTITUTABLE: replacing A with B or B with A at this exact boundary
   preserves the task-relevant local meaning, operation, or behavior established
   by the visible context.
2. BOTH_VALID_DIFFERENT: both candidates are locally coherent, but substituting
   one for the other changes the assertion, quantity, referent, reasoning
   direction, operation, or executable behavior.
3. A_UNSAFE: A is locally contradictory, syntactically invalid,
   mathematically invalid, or behaviorally unsafe according to the visible
   context, while B is locally coherent.
4. B_UNSAFE: B is locally unsafe according to the visible context, while A is
   locally coherent.
5. BOTH_UNSAFE: both candidates are locally invalid or contradictory according
   to the visible context.
6. INSUFFICIENT_BOUNDARY: the candidate token itself cannot yet be semantically
   or functionally resolved from the visible prefix, for example because it is
   a BPE fragment, incomplete identifier, incomplete operator, or otherwise
   necessarily depends on following tokens.

For LOCALLY_SUBSTITUTABLE, assign exactly one equivalence_basis:
- SEMANTIC: A and B preserve the same semantic content, proposition, referent,
  discourse meaning, or reasoning meaning. Lexical paraphrases such as
  "therefore" versus "thus" belong here.
- REPRESENTATIONAL: the equivalence is fully explained by a surface
  representation change, such as notation, punctuation, formatting,
  whitespace/tokenization, or an already-established consistent identifier
  renaming. If representational normalization alone explains the equivalence,
  use REPRESENTATIONAL rather than SEMANTIC.

Judge punctuation, whitespace, capitalization, and delimiters by their function
in the visible context. They are representationally substitutable when the
difference only changes presentation or non-task-relevant discourse
segmentation. They are different when the difference changes an established
grammatical construct, value, mathematical expression, code behavior, literal
content, or required delimiter balance.

For every relation other than LOCALLY_SUBSTITUTABLE, equivalence_basis must be
NOT_APPLICABLE.

Do not use INSUFFICIENT_BOUNDARY merely because the sentence, derivation, code
statement, or answer is unfinished. If the visible context already establishes
that A and B differ in meaning or operation, use BOTH_VALID_DIFFERENT even if
later text is unavailable.

A candidate is not unresolved merely because it could be extended into a
longer word, number, identifier, or expression. If the visible candidate tokens
already establish different lexical prefixes, complete words, numeric values,
operators, referents, punctuation functions, or syntactic roles, use
BOTH_VALID_DIFFERENT unless their difference is purely representational.

Examples:
- "therefore" versus "thus" is LOCALLY_SUBSTITUTABLE with SEMANTIC basis when
  both express the same inference.
- Capitalization, optional sentence punctuation, or newline-count differences
  are LOCALLY_SUBSTITUTABLE with REPRESENTATIONAL basis when they only alter
  presentation.
- The same kinds of punctuation or newline differences are
  BOTH_VALID_DIFFERENT when they alter an established list, code block, literal,
  delimiter structure, or executable behavior.
- Different directions, numeric values, variables, or arithmetic/comparison
  operators are BOTH_VALID_DIFFERENT when they change the established
  direction, quantity, referent, or operation.
- "approximately" versus "exactly" is BOTH_VALID_DIFFERENT when it changes the
  visible assertion.
- An incomplete fragment such as "re" may justify INSUFFICIENT_BOUNDARY only
  when its eventual lexical or operational role truly cannot be resolved
  without following tokens.

Use INSUFFICIENT_BOUNDARY only when at least one candidate cannot be assigned
any stable local lexical, syntactic, semantic, numeric, or operational role.
For this label, identify whether A, B, or BOTH depend on following tokens in
boundary_dependency and state exactly what following information is required.
For every other relation, boundary_dependency must be NONE.

Do not label candidates equivalent merely because both could begin valid
continuations. Both candidates being individually plausible or valid is not
sufficient for substitution safety.

A candidate may be labeled unsafe only when its invalidity follows from the
visible shared context itself. Do not solve ahead, infer unseen reasoning steps,
or use facts that require unseen continuation.

substitution_safe asks whether A can be replaced by B, or B by A, at this exact
boundary without changing the task-relevant local meaning, operation, or
behavior established by the visible context. The fact that both candidates are
individually plausible or locally valid is NOT sufficient for
substitution_safe=true.

substitution_safe must be the JSON boolean true only for
LOCALLY_SUBSTITUTABLE, the JSON boolean false for BOTH_VALID_DIFFERENT,
A_UNSAFE, B_UNSAFE, or BOTH_UNSAFE, and JSON null for INSUFFICIENT_BOUNDARY.
Do not return the strings "YES", "NO", or "UNPROVEN". Base the decision only on
visible evidence and return JSON only.
```

### User prompt

```text
Complete shared context immediately before the rejected token:
<CONTEXT>
{context}
</CONTEXT>

Candidate A (the branch ends immediately after this token):
<A>{a}</A>

Candidate B (the branch ends immediately after this token):
<B>{b}</B>

Return exactly one JSON object with keys:
a_local_status, b_local_status, relation, equivalence_basis,
boundary_dependency, required_following_information, same_syntactic_role,
same_semantic_role, same_operation_or_behavior, substitution_safe,
a_interpretation, b_interpretation, confidence, evidence.

a_local_status and b_local_status must be VALID, INVALID, or INSUFFICIENT.
confidence must be a number from 0 to 1. Do not infer which candidate is draft
or residual.
```

程序端在接受结果前强制检查关系、状态和安全标签的一致性：

```text
LOCALLY_SUBSTITUTABLE / BOTH_VALID_DIFFERENT
  -> A=VALID, B=VALID
A_UNSAFE
  -> A=INVALID, B=VALID
B_UNSAFE
  -> A=VALID, B=INVALID
BOTH_UNSAFE
  -> A=INVALID, B=INVALID
INSUFFICIENT_BOUNDARY
  -> 至少一边为 INSUFFICIENT
LOCALLY_SUBSTITUTABLE
  -> substitution_safe=true
LOCALLY_SUBSTITUTABLE
  -> equivalence_basis=SEMANTIC 或 REPRESENTATIONAL
其他关系
  -> equivalence_basis=NOT_APPLICABLE
INSUFFICIENT_BOUNDARY
  -> substitution_safe=null
  -> boundary_dependency=A、B或BOTH
  -> required_following_information非空
其余关系
  -> substitution_safe=false
  -> boundary_dependency=NONE
  -> required_following_information=null
```

任何冲突输出都会触发重试；连续重试仍不合法时才写入错误文件并跳过该样本。

## 5. 输入数据格式

输入 JSONL 每行保存一条抽样事件：

```json
{
  "schema_version": 1,
  "sample_index": 0,
  "sample_id": "lcb_v6:request_id:9624:1368:8938",
  "task_name": "lcb_v6",
  "request_id": "request_id",
  "generated_position": 9624,
  "context_text": "完整 prompt 与完整 generated prefix",
  "draft_token_id": 1368,
  "draft_token_text": " If",
  "residual_token_id": 8938,
  "residual_token_text": " Because",
  "source_event": {
    "table_frequency": 1,
    "table_hit": false,
    "draft_target_probability": 0.6486,
    "residual_target_probability": 0.2386,
    "entropy_raw": 0.8687
  }
}
```

Judge 脚本只把 `context_text` 和盲化后的两个 token 写入模型消息，其余字段不进入 Prompt。

## 6. 输出与指标

Judge 输出保留原始盲化判断和映射回 draft/residual 后的结果。核心安全集合为：

```text
LOCALLY_SUBSTITUTABLE
```

事件级安全替换率为：

```text
safe events / (all events - INSUFFICIENT_BOUNDARY events)
```

后续还应按有向 pair 聚合：

```text
pair_local_safe_rate = safe occurrences / decidable occurrences
```

并分析 calibration frequency、table hit、probability ratio、entropy、任务类型与
`pair_local_safe_rate` 的关系。frequency 不参与抽样或 Judge，只参与事后分析。

## 7. 运行入口

```bash
bash runs/mtp_csd/qwen35b_mtp314_rejection_trace/run_direct_local_judge_5pct.sh
```

脚本依次完成全局 5% 样本构造、DeepSeek-V4 Judge 服务启动、Direct Judge 和结果汇总。
所有步骤顺序执行；Judge 支持在相同输出目录下使用 `RESUME=1` 续跑。

单个样本发生 HTTP 超时、服务错误、JSON 解析错误或输出 schema 不合法时，会先按配置重试；
重试仍失败则写入 `judge_errors.jsonl` 并跳过该样本，其余任务继续运行。主结果文件只保存成功
判断。汇总文件同时报告期望样本数、成功数、失败记录数和不同失败样本数。使用
`RESUME=1` 重启时，已经成功的样本不会重复判断，先前失败的样本会重新尝试。
