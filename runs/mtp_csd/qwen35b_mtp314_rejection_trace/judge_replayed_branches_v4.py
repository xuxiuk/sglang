#!/usr/bin/env python3
"""V4 judge: independent validity with strict label-boundary constraints."""

import judge_replayed_branches_v3 as judge

judge.PROMPT_VERSION = "V4_BOUNDARY_CONSTRAINED_INDEPENDENT_VALIDITY"
base_validate = judge.validate
judge.SYSTEM = """You are a rigorous evaluator of a speculative-decoding rejection event.

You receive the complete shared context immediately before a one-token divergence and two deterministic counterfactual continuations. Their draft/residual identities are hidden. Evaluate A and B independently against the original task. Different reasoning methods may both be acceptable.

The purpose is to distinguish a demonstrated valid path, a coherent but not yet proven path, a concretely invalid path, and text that contains too little task-relevant reasoning to assess. Apply these boundaries exactly.

Rules:
1. PROVEN_VALID: the visible continuation contains sufficient task-relevant evidence establishing a correct and acceptable path. It need not use the same derivation as the other branch.
2. PLAUSIBLE_UNPROVEN: the continuation contains at least one coherent task-relevant reasoning step, algorithmic idea, derivation, invariant, construction, or partial implementation, and no concrete error is visible, but the visible continuation does not establish complete correctness.
3. INSUFFICIENT: use only when there is too little task-relevant content to identify or assess a reasoning direction. A missing final answer, truncation, or incompleteness alone is NOT sufficient for INSUFFICIENT.
4. PROVEN_INVALID: a specific independently verifiable mathematical, logical, factual, constraint, algorithmic, or executable-behavior error is visible.
5. Incompleteness alone is NEVER evidence of invalidity. An unfinished or truncated branch must not be PROVEN_INVALID unless you state a concrete error and verify why it is wrong.
6. If a branch has a coherent visible approach and no concrete error, classify it PLAUSIBLE_UNPROVEN even when it stops mid-sentence or lacks code, complexity analysis, edge cases, or a final answer.
7. Every PROVEN_INVALID branch must have a non-null concrete_error that identifies the erroneous claim or step and explains the contradiction or violated requirement. Do not write merely incomplete, truncated, unresolved, or no final answer.
8. Judge validity independently. Do not require A and B to share wording, intermediate steps, algorithm, proof technique, or final presentation.
9. same_correct_outcome=YES only if visible evidence establishes that both branches reach the same correct task-level result. Otherwise use NO or UNPROVEN.
10. path_relation=DISTINCT_BUT_VALID requires both branches to be PROVEN_VALID and to use materially different correct paths.
11. path_relation=EQUIVALENT requires both branches to be PROVEN_VALID, same_correct_outcome=YES, and task-relevant equivalence.
12. If either branch is not PROVEN_VALID, path_relation must be UNPROVEN.
13. For code, assess algorithm, constraints, edge cases, complexity requirements, and executable behavior. For mathematics, distinct valid derivations are allowed.
14. Ignore superficial wording, punctuation, formatting, notation, variable naming, and length differences.
15. Base positive validity and invalidity decisions only on visible evidence. Return JSON only."""


def validate_v4(verdict):
    base_validate(verdict)
    corrections = []
    a = verdict["a_validity"]
    b = verdict["b_validity"]
    relation = verdict["path_relation"]
    outcome = verdict["same_correct_outcome"]
    if a == "PROVEN_INVALID" and not verdict.get("a_concrete_error"):
        verdict["a_validity"] = "PLAUSIBLE_UNPROVEN"
        corrections.append("A_PROVEN_INVALID_WITHOUT_CONCRETE_ERROR_DOWNGRADED")
    if b == "PROVEN_INVALID" and not verdict.get("b_concrete_error"):
        verdict["b_validity"] = "PLAUSIBLE_UNPROVEN"
        corrections.append("B_PROVEN_INVALID_WITHOUT_CONCRETE_ERROR_DOWNGRADED")
    a = verdict["a_validity"]
    b = verdict["b_validity"]
    if relation == "DISTINCT_BUT_VALID" and (a != "PROVEN_VALID" or b != "PROVEN_VALID"):
        verdict["path_relation"] = "UNPROVEN"
        corrections.append("INCONSISTENT_DISTINCT_RELATION_DOWNGRADED")
    if relation == "EQUIVALENT" and (
        a != "PROVEN_VALID" or b != "PROVEN_VALID" or outcome != "YES"
    ):
        verdict["path_relation"] = "UNPROVEN"
        corrections.append("INCONSISTENT_EQUIVALENT_RELATION_DOWNGRADED")
    if (a != "PROVEN_VALID" or b != "PROVEN_VALID") and relation != "UNPROVEN":
        verdict["path_relation"] = "UNPROVEN"
        if "INCONSISTENT_DISTINCT_RELATION_DOWNGRADED" not in corrections and "INCONSISTENT_EQUIVALENT_RELATION_DOWNGRADED" not in corrections:
            corrections.append("NONVALID_BRANCH_RELATION_DOWNGRADED")
    verdict["v4_normalization_corrections"] = corrections


judge.validate = validate_v4

if __name__ == "__main__":
    judge.main()
