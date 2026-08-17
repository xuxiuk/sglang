#!/usr/bin/env python3
"""V3 multidimensional blind judge for replayed rejection branches."""

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

VALIDITIES = {"PROVEN_VALID", "PROVEN_INVALID", "PLAUSIBLE_UNPROVEN", "INSUFFICIENT"}
OUTCOMES = {"YES", "NO", "UNPROVEN"}
PATH_RELATIONS = {"EQUIVALENT", "DISTINCT_BUT_VALID", "UNPROVEN"}
PROMPT_VERSION = "V3_MULTIDIMENSIONAL_INDEPENDENT_VALIDITY"

SYSTEM = """You are a rigorous evaluator of a speculative-decoding rejection event.

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
12. Base every positive validity decision on visible evidence and identify that evidence in the output. Return JSON only."""

USER = """Shared context before the divergence:
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

Evaluate A and B independently. Return exactly one JSON object with keys:
a_validity, b_validity, same_correct_outcome, path_relation,
a_has_complete_answer, b_has_complete_answer, a_task_outcome, b_task_outcome,
a_validity_evidence, b_validity_evidence, a_concrete_error, b_concrete_error,
confidence, reason. Do not infer which branch is draft or residual."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--parallel-requests", type=int, default=64)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-ratio", type=float, default=1.0)
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


def post(url, headers, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def validate(verdict):
    if verdict.get("a_validity") not in VALIDITIES:
        raise ValueError(f"invalid a_validity: {verdict.get('a_validity')}")
    if verdict.get("b_validity") not in VALIDITIES:
        raise ValueError(f"invalid b_validity: {verdict.get('b_validity')}")
    if verdict.get("same_correct_outcome") not in OUTCOMES:
        raise ValueError(f"invalid same_correct_outcome: {verdict.get('same_correct_outcome')}")
    if verdict.get("path_relation") not in PATH_RELATIONS:
        raise ValueError(f"invalid path_relation: {verdict.get('path_relation')}")


def judge_one(index, row, cfg):
    event_key = f"{row['request_id']}:{row['generated_position']}:{index}"
    seed = int.from_bytes(hashlib.sha256(f"{cfg.seed}:{event_key}".encode()).digest()[:8], "big")
    swap = random.Random(seed).random() < 0.5
    draft = row["draft_branch"]["forced_token_text"] + row["draft_branch"]["continuation_text"]
    residual = row["residual_branch"]["forced_token_text"] + row["residual_branch"]["continuation_text"]
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
            response = post(cfg.base_url.rstrip("/") + "/chat/completions", headers, payload, cfg.timeout)
            verdict = parse_json(response["choices"][0]["message"]["content"])
            validate(verdict)
            draft_prefix = "a" if mapping["A"] == "draft" else "b"
            residual_prefix = "b" if draft_prefix == "a" else "a"
            mapped = {
                "draft_validity": verdict[f"{draft_prefix}_validity"],
                "residual_validity": verdict[f"{residual_prefix}_validity"],
                "draft_has_complete_answer": verdict.get(f"{draft_prefix}_has_complete_answer"),
                "residual_has_complete_answer": verdict.get(f"{residual_prefix}_has_complete_answer"),
                "draft_task_outcome": verdict.get(f"{draft_prefix}_task_outcome"),
                "residual_task_outcome": verdict.get(f"{residual_prefix}_task_outcome"),
                "draft_validity_evidence": verdict.get(f"{draft_prefix}_validity_evidence"),
                "residual_validity_evidence": verdict.get(f"{residual_prefix}_validity_evidence"),
                "draft_concrete_error": verdict.get(f"{draft_prefix}_concrete_error"),
                "residual_concrete_error": verdict.get(f"{residual_prefix}_concrete_error"),
                "same_correct_outcome": verdict["same_correct_outcome"],
                "path_relation": verdict["path_relation"],
            }
            return {
                "schema_version": 3,
                "prompt_version": PROMPT_VERSION,
                "event_index": index,
                "request_id": row["request_id"],
                "generated_position": row["generated_position"],
                "blind_mapping": mapping,
                "judge": verdict,
                "mapped_judgment": mapped,
                "source_event": row["source_event"],
                "judge_config": {"model": cfg.model, "temperature": cfg.temperature},
            }
        except Exception as exc:
            error = exc
            if attempt < cfg.max_retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"V3 judge failed for event {index}: {error}")


def main():
    cfg = parse_args()
    if cfg.sample_ratio != 1.0:
        raise ValueError("V3/V4 expect replay to perform sampling; --sample-ratio must be 1")
    if cfg.output.exists() and not cfg.resume:
        raise FileExistsError(f"Refusing to overwrite {cfg.output}")
    rows = [json.loads(line) for line in cfg.input.read_text().splitlines() if line]
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if cfg.output.exists():
        for line in cfg.output.read_text().splitlines():
            if line:
                result = json.loads(line)
                completed[int(result["event_index"])] = result
    pending = [(index, row) for index, row in enumerate(rows) if index not in completed]
    mode = "a" if cfg.output.exists() else "x"
    write_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.parallel_requests) as pool:
        futures = {pool.submit(judge_one, i, row, cfg): i for i, row in pending}
        with cfg.output.open(mode, encoding="utf-8", buffering=1) as handle:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with write_lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                completed[result["event_index"]] = result
    if len(completed) != len(rows):
        raise RuntimeError(f"V3 output incomplete: {len(completed)} of {len(rows)}")
    print(f"{PROMPT_VERSION} judged {len(completed)} events to {cfg.output}")


if __name__ == "__main__":
    main()
