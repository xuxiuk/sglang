#!/usr/bin/env python3
"""Replay one recorded GSM8K rejection per request to the final answer.

For every sampled rejection, generate two branches from the identical prefix:
one forces the rejected draft token and one forces the verifier residual token.
The resulting paired correctness directly measures the effect of one token
intervention. It does not model multiple force-accepts accumulated in one run.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from transformers import AutoTokenizer

from replay_rejection_branches import load_trace


ANSWER_RE = re.compile(r"ANSWER\s*:\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.I)
BOXED_RE = re.compile(r"\\boxed\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--model", default="/data/model/Qwen3.5-35B-A3B")
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--cuda-devices", default="0,1,2,3")
    p.add_argument("--mem-fraction-static", type=float, default=0.75)
    p.add_argument("--context-length", type=int, default=96000)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-running-requests", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Globally sample this fraction from all matched rejection events.",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional global cap after one-event-per-request sampling; default keeps all requests.",
    )
    p.add_argument(
        "--event-filter",
        choices=("all", "table-hit", "would-force-accept"),
        default="all",
    )
    p.add_argument(
        "--one-event-per-request",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.replace(",", "").strip().rstrip(".$")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if number == number.to_integral():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    if not matches:
        matches = BOXED_RE.findall(text)
    return normalize_number(matches[-1]) if matches else None


def load_gsm8k_gold() -> list[dict[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    return [
        {
            "question": row["question"],
            "gold": normalize_number(row["answer"].rsplit("####", 1)[-1]),
        }
        for row in dataset
    ]


def attach_gold(events: list[dict], tokenizer, gold_rows: list[dict]) -> None:
    cache: dict[str, tuple[str, str]] = {}
    for item in events:
        request_id = item["event"]["request_id"]
        if request_id not in cache:
            prompt = tokenizer.decode(item["request"]["prompt_token_ids"])
            matches = [row for row in gold_rows if row["question"] in prompt]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one GSM8K question in request {request_id}, got {len(matches)}"
                )
            cache[request_id] = (matches[0]["question"], matches[0]["gold"])
        item["question"], item["gold_answer"] = cache[request_id]


def sample_events(events: list[dict], args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.sample_seed)
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("--sample-ratio must be in (0, 1]")
    if args.one_event_per_request:
        by_request: dict[str, list[dict]] = {}
        for item in events:
            by_request.setdefault(item["event"]["request_id"], []).append(item)
        events = [rng.choice(items) for items in by_request.values()]
    rng.shuffle(events)
    sample_size = max(1, round(len(events) * args.sample_ratio))
    events = events[:sample_size]
    return events if args.max_events is None else events[: args.max_events]


def make_summary(records: list[dict]) -> dict:
    outcome = Counter(record["paired_outcome"] for record in records)
    n = len(records)
    draft_correct = sum(record["draft_correct"] for record in records)
    residual_correct = sum(record["residual_correct"] for record in records)
    return {
        "num_events": n,
        "paired_outcomes": dict(outcome),
        "both_correct_rate": outcome["both_correct"] / n if n else 0,
        "draft_only_correct_rate": outcome["draft_only_correct"] / n if n else 0,
        "residual_only_correct_harm_rate": outcome["residual_only_correct"] / n if n else 0,
        "both_wrong_rate": outcome["both_wrong"] / n if n else 0,
        "draft_branch_accuracy": draft_correct / n if n else 0,
        "residual_branch_accuracy": residual_correct / n if n else 0,
        "paired_accuracy_delta_draft_minus_residual": (
            (draft_correct - residual_correct) / n if n else 0
        ),
        "answer_changed_rate": (
            sum(r["draft_answer"] != r["residual_answer"] for r in records) / n
            if n
            else 0
        ),
        "draft_missing_answer_rate": (
            sum(r["draft_answer"] is None for r in records) / n if n else 0
        ),
        "residual_missing_answer_rate": (
            sum(r["residual_answer"] is None for r in records) / n if n else 0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("Refusing to overwrite counterfactual output")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    events = load_trace(args.trace_dir, args.event_filter)
    total_matched = len(events)
    events = sample_events(events, args)
    attach_gold(events, tokenizer, load_gsm8k_gold())
    if not events:
        raise ValueError("No events selected")

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    from sglang import Engine

    engine = Engine(
        model_path=args.model,
        tp_size=args.tp_size,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        trust_remote_code=True,
        skip_server_warmup=True,
        max_running_requests=args.max_running_requests,
    )
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "sampling_seed": args.seed,
        "stop": ["Question:"],
    }

    records: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for begin in range(0, len(events), args.batch_size):
            chunk = events[begin : begin + args.batch_size]
            inputs = []
            for item in chunk:
                event = item["event"]
                base = item["request"]["prompt_token_ids"] + item["generated_prefix"]
                inputs.extend(
                    [base + [event["draft_token_id"]], base + [event["residual_token_id"]]]
                )
            generated = engine.generate(input_ids=inputs, sampling_params=sampling_params)
            if len(generated) != 2 * len(chunk):
                raise RuntimeError("Unexpected number of branch outputs")

            for index, item in enumerate(chunk):
                event = item["event"]
                context_ids = (
                    item["request"]["prompt_token_ids"] + item["generated_prefix"]
                )
                prefix = tokenizer.decode(item["generated_prefix"])
                context_text = tokenizer.decode(context_ids)
                draft_tail = generated[2 * index]["text"]
                residual_tail = generated[2 * index + 1]["text"]
                draft_text = (
                    prefix
                    + tokenizer.decode([event["draft_token_id"]])
                    + draft_tail
                )
                residual_text = (
                    prefix
                    + tokenizer.decode([event["residual_token_id"]])
                    + residual_tail
                )
                draft_answer = extract_answer(draft_text)
                residual_answer = extract_answer(residual_text)
                draft_correct = draft_answer == item["gold_answer"]
                residual_correct = residual_answer == item["gold_answer"]
                if draft_correct and residual_correct:
                    paired_outcome = "both_correct"
                elif draft_correct:
                    paired_outcome = "draft_only_correct"
                elif residual_correct:
                    paired_outcome = "residual_only_correct"
                else:
                    paired_outcome = "both_wrong"
                record = {
                    "schema_version": 1,
                    "request_id": event["request_id"],
                    "generated_position": event["generated_position"],
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "context_token_ids": context_ids,
                    "context_text": context_text,
                    "source_event": event,
                    "draft_branch": {
                        "forced_token_id": event["draft_token_id"],
                        "forced_token_text": tokenizer.decode([event["draft_token_id"]]),
                        "continuation_text": draft_tail,
                        "finish_reason": generated[2 * index]
                        .get("meta_info", {})
                        .get("finish_reason"),
                    },
                    "residual_branch": {
                        "forced_token_id": event["residual_token_id"],
                        "forced_token_text": tokenizer.decode(
                            [event["residual_token_id"]]
                        ),
                        "continuation_text": residual_tail,
                        "finish_reason": generated[2 * index + 1]
                        .get("meta_info", {})
                        .get("finish_reason"),
                    },
                    "draft_answer": draft_answer,
                    "residual_answer": residual_answer,
                    "draft_correct": draft_correct,
                    "residual_correct": residual_correct,
                    "paired_outcome": paired_outcome,
                    "draft_completion": draft_text,
                    "residual_completion": residual_text,
                    "replay_config": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_new_tokens": args.max_new_tokens,
                        "seed": args.seed,
                        "single_intervention": True,
                    },
                }
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

    engine.shutdown()
    summary = make_summary(records)
    summary.update(
        {
            "trace_dir": str(args.trace_dir),
            "event_filter": args.event_filter,
            "total_matched_events": total_matched,
            "one_event_per_request": args.one_event_per_request,
            "sample_ratio": args.sample_ratio,
            "sample_seed": args.sample_seed,
            "selected_unique_requests": len({r["request_id"] for r in records}),
            "interpretation": "single forced-token intervention; not cumulative CSD",
        }
    )
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
