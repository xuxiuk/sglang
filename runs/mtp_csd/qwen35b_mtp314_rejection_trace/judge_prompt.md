# Counterfactual rejection judge prompt registry

This file retains every judge definition used or proposed for the rejection
trace experiment. Do not silently replace one version with another: results
must record the prompt version that produced them.

| Version | Name | Status | Intended use |
|---|---|---|---|
| V1 | Legacy lenient validity/equivalence judge | Historical; exact original text was not archived | Explain the first exploratory results only |
| V2 | Strict path-equivalence judge | Complete executable prompt | Conservative equivalence lower bound |
| V3 | Multidimensional independent-validity judge | Complete proposed prompt | Primary false-rejection analysis |

## V1 — Legacy lenient judge

The first completed run used the labels below:

```text
BOTH_VALID_EQUIVALENT
BOTH_VALID_DIFFERENT
DRAFT_BETTER
RESIDUAL_BETTER
BOTH_INVALID
INSUFFICIENT_CONTEXT
```

Its effective decision rule treated two locally coherent continuations without
an observed error as valid too readily. This produced 6,670
`BOTH_VALID_EQUIVALENT` events out of 9,645 judged events. The exact original
System/User strings were overwritten before they were independently archived,
so they must not be represented as a verbatim reproducible prompt. Historical
outputs remain under `runs/full_pipeline_20260813_015303/*/judge/judged.jsonl`.

V1 is retained here as provenance, not as the final false-rejection definition.

## V2 — Strict path-equivalence judge

This is the complete prompt used by
`judge_replayed_branches.py`. Before constructing the prompt, the script
deterministically randomizes whether the draft branch is shown as A or B. The
judge therefore cannot use the draft/residual identity when making its decision.

## System message

```text
You are a strict evaluator of a speculative-decoding rejection event.

You receive the complete shared context immediately before a one-token divergence and two deterministic continuations. Their draft/residual identities are hidden.

Your task is not to decide whether both continuations merely sound fluent or could possibly become correct. Determine whether the supplied text provides concrete evidence that rejecting one branch was unnecessary.

Evaluate each branch independently against the original task in the context. Check mathematical reasoning, factual claims, algorithmic logic, code behavior, constraints, and final answers when available.

Strict rules:
1. Use PROVEN_EQUIVALENT only when the supplied continuations themselves establish that both branches preserve the same task-relevant semantics and reach the same correct conclusion, equivalent valid derivation, or behaviorally equivalent code.
2. Local fluency, grammatical validity, or absence of an immediately visible error is not evidence of equivalence.
3. If both branches remain plausible but follow unfinished or different derivations, use BOTH_PLAUSIBLE_UNPROVEN.
4. If neither branch reaches enough task-relevant substance to compare, use INSUFFICIENT_CONTEXT.
5. If one branch has a concrete logical, mathematical, factual, or behavioral error and the other does not, use A_BETTER or B_BETTER.
6. Do not penalize superficial wording, formatting, punctuation, notation, variable naming, or length differences.
7. For code, compare algorithm and executable behavior. If equivalence cannot be established, do not claim it.
8. For mathematics, the same final answer is insufficient if one derivation is invalid.
9. Never assume an unfinished branch will repair itself later.
10. Base the judgment only on visible evidence. Return JSON only.
```

## User message template

```text
Shared context before the divergence:
<CONTEXT>
{context}
</CONTEXT>

Continuation A:
<A>
{a}
</A>

Continuation B:
<B>
{b}
</B>

Return exactly one JSON object with keys label, a_status, b_status, task_type,
a_conclusion, b_conclusion, same_conclusion, a_concrete_error,
b_concrete_error, equivalence_evidence, confidence, reason. label must be one of:
PROVEN_EQUIVALENT, BOTH_PLAUSIBLE_UNPROVEN, A_BETTER, B_BETTER,
BOTH_INVALID, INSUFFICIENT_CONTEXT.
```

## Expected JSON schema

```json
{
  "label": "PROVEN_EQUIVALENT | BOTH_PLAUSIBLE_UNPROVEN | A_BETTER | B_BETTER | BOTH_INVALID | INSUFFICIENT_CONTEXT",
  "a_status": "VALID | INVALID | UNPROVEN",
  "b_status": "VALID | INVALID | UNPROVEN",
  "task_type": "CODE | MATH | GENERAL",
  "a_conclusion": null,
  "b_conclusion": null,
  "same_conclusion": false,
  "a_concrete_error": null,
  "b_concrete_error": null,
  "equivalence_evidence": null,
  "confidence": 0.0,
  "reason": "brief evidence-based explanation"
}
```

## Post-processing semantics

The stored `blind_mapping` records which hidden branch was assigned to A and B.
After judging:

- `A_BETTER` and `B_BETTER` are mapped back to `DRAFT_BETTER` or
  `RESIDUAL_BETTER`;
- `PROVEN_EQUIVALENT` is retained unchanged;
- strict evidence of an unnecessary rejection is counted only when the mapped
  label is `PROVEN_EQUIVALENT` or `DRAFT_BETTER`;
- `BOTH_PLAUSIBLE_UNPROVEN` is reported separately and is not counted as strict
  evidence.

V2 measures a conservative path-equivalence lower bound. It can undercount
unnecessary rejections when draft and residual continuations follow distinct
but independently correct solution paths.

## V3 — Multidimensional independent-validity judge

V3 separates branch validity, outcome agreement, and path equivalence. It does
not require the draft branch to use the same derivation as the residual branch.
The A/B presentation remains blinded and is mapped back only after judging.

### System message

```text
You are a rigorous evaluator of a speculative-decoding rejection event.

You receive the complete shared context immediately before a one-token divergence and two deterministic counterfactual continuations. Their draft/residual identities are hidden.

Evaluate Continuation A and Continuation B independently against the original task. The central question is whether either continuation is demonstrated by the visible evidence to be a correct and acceptable solution path. Two branches may both be valid even when they use different reasoning methods, intermediate representations, wording, algorithms, or code structure.

Do not equate local fluency, grammatical validity, or the absence of an immediately visible error with a proven-valid solution path.

Rules:
1. Assign PROVEN_VALID only when the supplied continuation itself contains enough task-relevant evidence to establish a correct and acceptable path. A complete correct answer is strong evidence. A rigorous partial derivation may qualify only if the visible segment already establishes the disputed step and contains no unresolved dependency on unseen future repair.
2. Assign PROVEN_INVALID when a concrete mathematical, logical, factual, constraint, algorithmic, or executable-behavior error is visible.
3. Assign PLAUSIBLE_UNPROVEN when a branch appears locally reasonable but is unfinished or lacks enough evidence to establish correctness.
4. Assign INSUFFICIENT when the visible material is too truncated or irrelevant even to assess plausibility.
5. Judge A and B independently. Do not require the two branches to share a derivation or surface form.
6. Set same_correct_outcome=YES only when the visible evidence establishes that both branches reach the same correct task-level answer or behavior. Use UNPROVEN when either outcome is unfinished or unclear.
7. Set path_relation=EQUIVALENT only when the branches are task-relevantly equivalent, not merely fluent. Set DISTINCT_BUT_VALID when both are PROVEN_VALID but use materially different valid reasoning paths or implementations. Otherwise use UNPROVEN.
8. For code, evaluate algorithm, constraints, edge cases, complexity requirements, and executable behavior. Different implementations may both be valid.
9. For mathematics, different derivations may both be valid. Do not require algebraic and geometric proofs, for example, to be path-equivalent.
10. Never assume an unfinished branch will fix an error or eventually reach the correct answer.
11. Ignore superficial differences in wording, formatting, punctuation, notation, variable names, and response length.
12. Base every positive validity decision on visible evidence and identify that evidence in the output. Return JSON only.
```

### User message template

```text
Shared context before the divergence:
<CONTEXT>
{context}
</CONTEXT>

Continuation A:
<A>
{a}
</A>

Continuation B:
<B>
{b}
</B>

Evaluate A and B independently. Return exactly one JSON object matching the requested schema. Do not infer which branch is draft or residual.
```

### Expected JSON schema

```json
{
  "a_validity": "PROVEN_VALID | PROVEN_INVALID | PLAUSIBLE_UNPROVEN | INSUFFICIENT",
  "b_validity": "PROVEN_VALID | PROVEN_INVALID | PLAUSIBLE_UNPROVEN | INSUFFICIENT",
  "same_correct_outcome": "YES | NO | UNPROVEN",
  "path_relation": "EQUIVALENT | DISTINCT_BUT_VALID | UNPROVEN",
  "a_has_complete_answer": false,
  "b_has_complete_answer": false,
  "a_task_outcome": null,
  "b_task_outcome": null,
  "a_validity_evidence": null,
  "b_validity_evidence": null,
  "a_concrete_error": null,
  "b_concrete_error": null,
  "confidence": 0.0,
  "reason": "brief evidence-based comparison"
}
```

### Metrics derived after blind mapping

After mapping A/B back to draft/residual, report these metrics separately:

```text
validated_draft_rate
  = count(draft_validity == PROVEN_VALID) / all judged events

strong_draft_only_evidence_rate
  = count(draft_validity == PROVEN_VALID
          and residual_validity == PROVEN_INVALID) / all judged events

both_valid_rate
  = count(draft_validity == PROVEN_VALID
          and residual_validity == PROVEN_VALID) / all judged events

distinct_valid_path_rate
  = count(both branches PROVEN_VALID
          and path_relation == DISTINCT_BUT_VALID) / all judged events

equivalent_path_lower_bound
  = count(both branches PROVEN_VALID
          and same_correct_outcome == YES
          and path_relation == EQUIVALENT) / all judged events

plausible_unproven_draft_rate
  = count(draft_validity == PLAUSIBLE_UNPROVEN) / all judged events
```

`validated_draft_rate` is the primary V3 signal. It should be described as a
counterfactually validated draft-path rate, not as an absolute ground-truth
false-rejection rate, because replay observes a finite continuation rather than
every possible future trajectory.

## V4 — Boundary-constrained independent-validity judge

V4 uses the same blinded User message and JSON schema as V3, but replaces the
System message with the boundary-constrained prompt stored verbatim in
`judge_replayed_branches_v4.py`. Its additional mandatory rules are:

```text
1. PROVEN_VALID requires visible evidence establishing a correct acceptable path.
2. PLAUSIBLE_UNPROVEN applies whenever at least one coherent task-relevant step,
   algorithmic idea, derivation, invariant, construction, or partial implementation
   is visible, no concrete error is visible, and complete correctness is not yet established.
3. INSUFFICIENT is reserved for text with too little task-relevant content even to
   identify or assess a reasoning direction. Missing final answers, truncation, and
   incompleteness alone do not imply INSUFFICIENT.
4. PROVEN_INVALID requires a specific independently verifiable error.
5. Incompleteness alone is never evidence of invalidity.
6. Every PROVEN_INVALID response must provide a non-null concrete_error identifying
   the erroneous step and explaining the contradiction or violated requirement.
7. DISTINCT_BUT_VALID requires both branches to be PROVEN_VALID.
8. EQUIVALENT requires both branches to be PROVEN_VALID and same_correct_outcome=YES.
9. If either branch is not PROVEN_VALID, path_relation must be UNPROVEN.
```

These relationship constraints are also validated in Python. Structurally
inconsistent responses are rejected and retried rather than included in the
result. V4 defines candidate false rejections as:

```text
draft_validity in {PROVEN_VALID, PLAUSIBLE_UNPROVEN}
```

The two components must still be reported separately so that the candidate
rate is not misrepresented as a proven ground-truth false-rejection rate.
