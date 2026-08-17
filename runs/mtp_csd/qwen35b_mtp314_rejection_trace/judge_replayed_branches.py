#!/usr/bin/env python3
"""Blindly judge replayed draft/residual branches through an OpenAI endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

LABELS = {
    "PROVEN_EQUIVALENT",
    "BOTH_PLAUSIBLE_UNPROVEN",
    "A_BETTER",
    "B_BETTER",
    "BOTH_INVALID",
    "INSUFFICIENT_CONTEXT",
}
SYSTEM = """You are a strict evaluator of a speculative-decoding rejection event.

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
10. Base the judgment only on visible evidence. Return JSON only."""
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

Return exactly one JSON object with keys label, a_status, b_status, task_type,
a_conclusion, b_conclusion, same_conclusion, a_concrete_error,
b_concrete_error, equivalence_evidence, confidence, reason. label must be one of:
PROVEN_EQUIVALENT, BOTH_PLAUSIBLE_UNPROVEN, A_BETTER, B_BETTER,
BOTH_INVALID, INSUFFICIENT_CONTEXT."""


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--parallel-requests", type=int, default=32)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Deterministically sample this fraction of replay rows before judging.",
    )
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
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def judge_one(index, row, cfg):
    event_key = f"{row['request_id']}:{row['generated_position']}:{index}"
    local_seed = int.from_bytes(
        hashlib.sha256(f"{cfg.seed}:{event_key}".encode()).digest()[:8], "big"
    )
    swap = random.Random(local_seed).random() < 0.5
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
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"}
    last_error = None
    for attempt in range(cfg.max_retries + 1):
        try:
            response = post(url, headers, payload, cfg.timeout)
            raw = response["choices"][0]["message"]["content"]
            verdict = parse_json(raw)
            if verdict.get("label") not in LABELS:
                raise ValueError(f"Invalid judge label: {verdict.get('label')}")
            label = verdict["label"]
            if label == "A_BETTER":
                mapped = mapping["A"].upper() + "_BETTER"
            elif label == "B_BETTER":
                mapped = mapping["B"].upper() + "_BETTER"
            else:
                mapped = label
            return {
                "schema_version": 1,
                "event_index": index,
                "request_id": row["request_id"],
                "generated_position": row["generated_position"],
                "blind_mapping": mapping,
                "judge": verdict,
                "mapped_label": mapped,
                "source_event": row["source_event"],
                "judge_config": {"model": cfg.model, "temperature": cfg.temperature},
            }
        except Exception as exc:
            last_error = exc
            if attempt < cfg.max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge failed for event {index}: {last_error}")


def main():
    cfg = args()
    if cfg.output.exists():
        raise FileExistsError(f"Refusing to overwrite {cfg.output}")
    rows = [json.loads(x) for x in cfg.input.read_text().splitlines() if x]
    if not 0 < cfg.sample_ratio <= 1:
        raise ValueError("--sample-ratio must be in (0, 1]")
    total_replay_events = len(rows)
    if cfg.sample_ratio < 1 and rows:
        sample_size = max(1, round(total_replay_events * cfg.sample_ratio))
        selected = sorted(random.Random(cfg.seed).sample(range(total_replay_events), sample_size))
        rows = [rows[index] for index in selected]
    else:
        selected = list(range(total_replay_events))
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    results = [None] * len(rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.parallel_requests) as pool:
        futures = {pool.submit(judge_one, i, row, cfg): i for i, row in enumerate(rows)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result["replay_event_index"] = selected[result["event_index"]]
            result["judge_sampling"] = {
                "sample_ratio": cfg.sample_ratio,
                "sample_seed": cfg.seed,
                "total_replay_events": total_replay_events,
                "sampled_events": len(rows),
            }
            results[result["event_index"]] = result
    with cfg.output.open("x", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Judged {len(results)} events to {cfg.output}")


if __name__ == "__main__":
    main()
