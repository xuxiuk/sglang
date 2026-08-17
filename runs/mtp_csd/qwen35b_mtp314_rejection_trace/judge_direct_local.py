#!/usr/bin/env python3
"""Blind direct judge for local draft/residual token substitution."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import threading
import time
import urllib.request
from pathlib import Path

PROMPT_VERSION = "V5_5_1_RESOLVE_THEN_COMPARE"
STATUSES = {"VALID", "INVALID", "INSUFFICIENT"}
RELATIONS = {
    "LOCALLY_SUBSTITUTABLE",
    "BOTH_VALID_DIFFERENT",
    "A_UNSAFE",
    "B_UNSAFE",
    "BOTH_UNSAFE",
    "INSUFFICIENT_BOUNDARY",
}
EQUIVALENCE_BASES = {"SEMANTIC", "REPRESENTATIONAL", "NOT_APPLICABLE"}
BOUNDARY_DEPENDENCIES = {"NONE", "A", "B", "BOTH"}
TASK_EFFECT_CATEGORIES = {
    "NONE",
    "PROPOSITION",
    "REFERENT",
    "QUANTITY",
    "OPERATION",
    "REASONING_DIRECTION",
    "EXECUTABLE_BEHAVIOR",
    "REQUIRED_STRUCTURE",
    "INVALIDITY",
    "UNRESOLVED",
}

SYSTEM = """You are a rigorous evaluator of a blind local token substitution.

You receive the complete shared context immediately before a one-token divergence and two candidate tokens A and B. Their draft/residual identities are hidden. Each branch consists exactly of the shared context followed by its candidate token, and ends immediately after that token. No continuation is available.

Judge only the relationship established at this exact position. Do not predict, imagine, or generate a continuation. Do not decide whether either branch will eventually reach the correct final answer.

Follow this order and do not skip step 1.

Step 1 — resolve each candidate token itself. A candidate is resolved if the visible token already has any stable lexical, numeric, syntactic, punctuation, delimiter, mathematical, or operational contribution. A sentence or phrase does not need to be complete. A verb needing an object, an adjective needing a noun, a determiner needing a noun phrase, a number allowing later digits, an operator needing an operand, or a delimiter needing a matching delimiter does NOT make the visible token unresolved. A word or subword prefix also has a stable visible lexical contribution even if more characters could follow.

INSUFFICIENT_BOUNDARY is reserved only for a decoded candidate that is empty, corrupted, unreadable, or otherwise has no identifiable local surface contribution at all. If the displayed candidate contains any readable character, word fragment, whitespace, punctuation, delimiter, digit, variable, or operator, it is resolved and INSUFFICIENT_BOUNDARY is forbidden.

If both candidates are resolved, INSUFFICIENT_BOUNDARY is forbidden. Compare them in step 2 even when the sentence, derivation, or code is unfinished.

Step 2 — compare task-relevant effect. Ask whether replacing one visible token with the other changes a proposition, referent, quantity, mathematical/logical operation, reasoning direction, executable behavior, literal content, or structure required by the visible context. A grammatical or discourse-form difference alone is not a task-relevant change. Surface presentation, punctuation, capitalization, whitespace, tokenization, or sentence segmentation may be representationally substitutable when they do not cause one of the concrete effects above. Both candidates merely being plausible is not sufficient for substitutability.

Use exactly one relation label:
1. LOCALLY_SUBSTITUTABLE: replacing A with B or B with A at this exact boundary preserves the task-relevant local meaning, operation, or behavior established by the visible context.
2. BOTH_VALID_DIFFERENT: both candidates are locally coherent, but substituting one for the other changes the assertion, quantity, referent, reasoning direction, operation, or executable behavior.
3. A_UNSAFE: A is locally contradictory, syntactically invalid, mathematically invalid, or behaviorally unsafe according to the visible context, while B is locally coherent.
4. B_UNSAFE: B is locally unsafe according to the visible context, while A is locally coherent.
5. BOTH_UNSAFE: both candidates are locally invalid or contradictory according to the visible context.
6. INSUFFICIENT_BOUNDARY: the candidate token itself cannot yet be semantically or functionally resolved from the visible prefix, for example because it is a BPE fragment, incomplete identifier, incomplete operator, or otherwise necessarily depends on following tokens.

For LOCALLY_SUBSTITUTABLE, assign exactly one equivalence_basis:
- SEMANTIC: A and B preserve the same semantic content, proposition, referent, discourse meaning, or reasoning meaning. Lexical paraphrases such as "therefore" versus "thus" belong here.
- REPRESENTATIONAL: the equivalence is fully explained by a surface representation change, such as notation, punctuation, formatting, whitespace/tokenization, or an already-established consistent identifier renaming. If representational normalization alone explains the equivalence, use REPRESENTATIONAL rather than SEMANTIC.

For every verdict, assign task_effect_category and concrete_task_effect:
- LOCALLY_SUBSTITUTABLE requires task_effect_category=NONE and concrete_task_effect="NONE".
- BOTH_VALID_DIFFERENT requires one of PROPOSITION, REFERENT, QUANTITY, OPERATION, REASONING_DIRECTION, EXECUTABLE_BEHAVIOR, or REQUIRED_STRUCTURE, plus a concrete description of what changes. "Different syntax", "different discourse function", or "one continues and one ends the sentence" is not a concrete task effect.
- A_UNSAFE, B_UNSAFE, or BOTH_UNSAFE requires INVALIDITY and the visible contradiction or invalid behavior.
- INSUFFICIENT_BOUNDARY requires UNRESOLVED and must explain why the displayed token itself is empty, corrupted, or unreadable. Missing words, operands, suffixes, or closing delimiters do not qualify.

For every relation other than LOCALLY_SUBSTITUTABLE, equivalence_basis must be NOT_APPLICABLE.

Examples:
- "therefore" versus "thus" is LOCALLY_SUBSTITUTABLE with SEMANTIC basis when both express the same inference.
- Capitalization, optional sentence punctuation, or newline-count differences are LOCALLY_SUBSTITUTABLE with REPRESENTATIONAL basis when they only alter presentation.
- Punctuation or newline differences are BOTH_VALID_DIFFERENT when they alter an established argument list, code block, literal, delimiter structure, or executable behavior.
- Different directions, numeric values, variables, or arithmetic/comparison operators are BOTH_VALID_DIFFERENT when they change an established direction, quantity, referent, or operation.
- A readable fragment such as "re", a standalone operator such as "=", and an opening quote or math delimiter are resolved local contributions even though later characters are needed to complete a word, expression, or literal.

For INSUFFICIENT_BOUNDARY, identify whether A, B, or BOTH depend on following characters in boundary_dependency and state exactly what characters or token identity information is required. For every other relation, boundary_dependency must be NONE.

Do not label candidates equivalent merely because both could begin valid continuations. Both candidates being individually plausible or valid is not sufficient for substitution safety.

A candidate may be labeled unsafe only when its invalidity follows from the visible shared context itself. Do not solve ahead, infer unseen reasoning steps, or use facts that require unseen continuation.

same_syntactic_role is diagnostic only and must not determine relation by itself.

substitution_safe must be the JSON boolean true only for LOCALLY_SUBSTITUTABLE, the JSON boolean false for BOTH_VALID_DIFFERENT, A_UNSAFE, B_UNSAFE, or BOTH_UNSAFE, and JSON null for INSUFFICIENT_BOUNDARY. Do not return the strings "YES", "NO", or "UNPROVEN". Base the decision only on visible evidence and return JSON only."""

USER = """Complete shared context immediately before the rejected token:
<CONTEXT>
{context}
</CONTEXT>

Candidate A (the branch ends immediately after this token):
<A>{a}</A>

Candidate B (the branch ends immediately after this token):
<B>{b}</B>

Return exactly one JSON object with keys:
a_local_status, b_local_status, relation, equivalence_basis, boundary_dependency,
required_following_information, task_effect_category, concrete_task_effect,
same_syntactic_role,
same_semantic_role, same_operation_or_behavior, substitution_safe,
a_interpretation, b_interpretation, confidence, evidence.

a_local_status and b_local_status must be VALID, INVALID, or INSUFFICIENT.
confidence must be a number from 0 to 1. Do not infer which candidate is draft or residual."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--error-output",
        type=Path,
        help="Write permanently failed samples here and continue judging.",
    )
    p.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--parallel-requests", type=int, default=64)
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def validate(v, a=None, b=None):
    relation = v.get("relation")
    if v.get("a_local_status") not in STATUSES or v.get("b_local_status") not in STATUSES:
        raise ValueError("invalid local status")
    if relation not in RELATIONS:
        raise ValueError(f"invalid relation: {relation}")
    if v.get("equivalence_basis") not in EQUIVALENCE_BASES:
        raise ValueError(f"invalid equivalence_basis: {v.get('equivalence_basis')}")
    if v.get("boundary_dependency") not in BOUNDARY_DEPENDENCIES:
        raise ValueError(f"invalid boundary_dependency: {v.get('boundary_dependency')}")
    effect_category = v.get("task_effect_category")
    if effect_category not in TASK_EFFECT_CATEGORIES:
        raise ValueError(f"invalid task_effect_category: {effect_category}")
    concrete_effect = str(v.get("concrete_task_effect") or "").strip()
    expected = (
        True if v["relation"] == "LOCALLY_SUBSTITUTABLE"
        else None if v["relation"] == "INSUFFICIENT_BOUNDARY"
        else False
    )
    if v.get("substitution_safe") is not expected:
        raise ValueError(
            f"inconsistent relation/safety: {v['relation']} vs {v['substitution_safe']}"
        )
    if relation == "LOCALLY_SUBSTITUTABLE":
        if v["equivalence_basis"] not in {"SEMANTIC", "REPRESENTATIONAL"}:
            raise ValueError("LOCALLY_SUBSTITUTABLE requires a concrete equivalence_basis")
        if effect_category != "NONE" or concrete_effect != "NONE":
            raise ValueError("LOCALLY_SUBSTITUTABLE requires task effect NONE")
    elif v["equivalence_basis"] != "NOT_APPLICABLE":
        raise ValueError(f"{relation} requires equivalence_basis=NOT_APPLICABLE")
    if relation == "BOTH_VALID_DIFFERENT":
        allowed = {
            "PROPOSITION", "REFERENT", "QUANTITY", "OPERATION",
            "REASONING_DIRECTION", "EXECUTABLE_BEHAVIOR", "REQUIRED_STRUCTURE",
        }
        if effect_category not in allowed or not concrete_effect or concrete_effect == "NONE":
            raise ValueError("BOTH_VALID_DIFFERENT requires a concrete task effect")
    if relation in {"A_UNSAFE", "B_UNSAFE", "BOTH_UNSAFE"}:
        if effect_category != "INVALIDITY" or not concrete_effect or concrete_effect == "NONE":
            raise ValueError(f"{relation} requires a concrete INVALIDITY effect")
    if relation == "INSUFFICIENT_BOUNDARY":
        if effect_category != "UNRESOLVED" or not concrete_effect or concrete_effect == "NONE":
            raise ValueError("INSUFFICIENT_BOUNDARY requires UNRESOLVED task effect")
        if a is not None and b is not None:
            readable = lambda token: bool(token) and "\ufffd" not in token
            if readable(a) and readable(b):
                raise ValueError(
                    "INSUFFICIENT_BOUNDARY forbidden: both displayed tokens have readable surface contributions"
                )
        if v["boundary_dependency"] == "NONE":
            raise ValueError("INSUFFICIENT_BOUNDARY requires a concrete boundary_dependency")
        if not str(v.get("required_following_information") or "").strip():
            raise ValueError(
                "INSUFFICIENT_BOUNDARY requires required_following_information"
            )
    else:
        if v["boundary_dependency"] != "NONE":
            raise ValueError(f"{relation} requires boundary_dependency=NONE")
        if v.get("required_following_information") not in {None, "", "NONE"}:
            raise ValueError(
                f"{relation} requires null/empty/NONE required_following_information"
            )
    a_status = v["a_local_status"]
    b_status = v["b_local_status"]
    expected_statuses = {
        "LOCALLY_SUBSTITUTABLE": ("VALID", "VALID"),
        "BOTH_VALID_DIFFERENT": ("VALID", "VALID"),
        "A_UNSAFE": ("INVALID", "VALID"),
        "B_UNSAFE": ("VALID", "INVALID"),
        "BOTH_UNSAFE": ("INVALID", "INVALID"),
    }
    if relation in expected_statuses and (a_status, b_status) != expected_statuses[relation]:
        raise ValueError(
            f"inconsistent relation/statuses: {relation} vs {(a_status, b_status)}"
        )
    if relation == "INSUFFICIENT_BOUNDARY" and "INSUFFICIENT" not in {a_status, b_status}:
        raise ValueError(
            "INSUFFICIENT_BOUNDARY requires at least one INSUFFICIENT local status"
        )
    for field in (
        "same_syntactic_role",
        "same_semantic_role",
        "same_operation_or_behavior",
    ):
        if not isinstance(v.get(field), bool):
            raise ValueError(f"{field} must be boolean")
    confidence = float(v.get("confidence"))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0,1]")


def post(url, headers, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def judge_one(index, row, cfg):
    digest = hashlib.sha256(f"{cfg.seed}:{row['sample_id']}".encode()).digest()
    swap = random.Random(int.from_bytes(digest[:8], "big")).random() < 0.5
    draft = row["draft_token_text"]
    residual = row["residual_token_text"]
    a, b = (residual, draft) if swap else (draft, residual)
    mapping = {"A": "residual", "B": "draft"} if swap else {"A": "draft", "B": "residual"}
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(context=row["context_text"], a=a, b=b)},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"}
    error = None
    for attempt in range(cfg.max_retries + 1):
        try:
            response = post(
                cfg.base_url.rstrip("/") + "/chat/completions",
                headers,
                payload,
                cfg.timeout,
            )
            verdict = parse_json(response["choices"][0]["message"]["content"])
            validate(verdict, a, b)
            draft_side = "a" if mapping["A"] == "draft" else "b"
            residual_side = "b" if draft_side == "a" else "a"
            mapped = {
                "draft_local_status": verdict[f"{draft_side}_local_status"],
                "residual_local_status": verdict[f"{residual_side}_local_status"],
                "draft_interpretation": verdict.get(f"{draft_side}_interpretation"),
                "residual_interpretation": verdict.get(f"{residual_side}_interpretation"),
                "relation": verdict["relation"],
                "equivalence_basis": verdict["equivalence_basis"],
                "boundary_dependency": verdict["boundary_dependency"],
                "required_following_information": verdict.get(
                    "required_following_information"
                ),
                "task_effect_category": verdict["task_effect_category"],
                "concrete_task_effect": verdict["concrete_task_effect"],
                "substitution_safe": verdict["substitution_safe"],
            }
            return {
                "schema_version": 1,
                "prompt_version": PROMPT_VERSION,
                "sample_index": row["sample_index"],
                "sample_id": row["sample_id"],
                "task_name": row["task_name"],
                "request_id": row["request_id"],
                "generated_position": row["generated_position"],
                "blind_mapping": mapping,
                "judge": verdict,
                "mapped_judgment": mapped,
                "draft_token_id": row["draft_token_id"],
                "draft_token_text": row["draft_token_text"],
                "residual_token_id": row["residual_token_id"],
                "residual_token_text": row["residual_token_text"],
                "source_event": row["source_event"],
                "judge_config": {"model": cfg.model, "temperature": cfg.temperature},
            }
        except Exception as exc:
            error = exc
            if attempt < cfg.max_retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Direct local judge failed for sample {row['sample_id']}: {error}")


def main():
    cfg = parse_args()
    if cfg.error_output is None:
        cfg.error_output = cfg.output.with_name(cfg.output.stem + ".errors.jsonl")
    if cfg.output.exists() and not cfg.resume:
        raise FileExistsError(f"Refusing to overwrite {cfg.output}")
    if cfg.error_output.exists() and not cfg.resume:
        raise FileExistsError(f"Refusing to overwrite {cfg.error_output}")
    with cfg.input.open(encoding="utf-8") as input_handle:
        rows = [json.loads(line) for line in input_handle if line.strip()]
    completed = {}
    if cfg.output.exists():
        for line in cfg.output.read_text().splitlines():
            if line:
                item = json.loads(line)
                completed[int(item["sample_index"])] = item
    pending = [(i, row) for i, row in enumerate(rows) if row["sample_index"] not in completed]
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if cfg.output.exists() else "x"
    error_mode = "a" if cfg.error_output.exists() else "x"
    lock = threading.Lock()
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.parallel_requests) as pool:
        futures = {pool.submit(judge_one, i, row, cfg): i for i, row in pending}
        with (
            cfg.output.open(mode, encoding="utf-8", buffering=1) as handle,
            cfg.error_output.open(error_mode, encoding="utf-8", buffering=1) as error_handle,
        ):
            for future in concurrent.futures.as_completed(futures):
                pending_index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures += 1
                    row = rows[pending_index]
                    failure = {
                        "schema_version": 1,
                        "prompt_version": PROMPT_VERSION,
                        "sample_index": row["sample_index"],
                        "sample_id": row["sample_id"],
                        "task_name": row["task_name"],
                        "request_id": row["request_id"],
                        "generated_position": row["generated_position"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "attempts": cfg.max_retries + 1,
                    }
                    with lock:
                        error_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        error_handle.flush()
                    continue
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                completed[result["sample_index"]] = result
    print(
        f"{PROMPT_VERSION}: input={len(rows)}, success={len(completed)}, "
        f"failed_this_run={failures}, output={cfg.output}, errors={cfg.error_output}"
    )


if __name__ == "__main__":
    main()
