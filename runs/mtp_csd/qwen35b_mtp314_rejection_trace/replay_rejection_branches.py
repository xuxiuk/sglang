#!/usr/bin/env python3
"""Replay both continuations of recorded MTP rejection events with SGLang."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/data/model/Qwen3.5-35B-A3B")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--cuda-devices", default="0,1,2,3")
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--context-length", type=int, default=96000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Globally sample this fraction of matched rejection events before replay.",
    )
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--event-filter",
        choices=("all", "table-hit", "would-force-accept"),
        default="table-hit",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_trace(trace_dir: Path, event_filter: str) -> list[dict]:
    requests: dict[str, dict] = {}
    for path in sorted(trace_dir.glob("requests.rank*.jsonl")):
        for record in read_jsonl(path):
            requests[record["request_id"]] = record

    selected: list[dict] = []
    seen: set[tuple] = set()
    skipped_unreplayable = 0
    for path in sorted(trace_dir.glob("events.rank*.jsonl")):
        for event in read_jsonl(path):
            if event_filter == "table-hit" and not event["table_hit"]:
                continue
            if event_filter == "would-force-accept" and not event["would_force_accept"]:
                continue
            key = (
                event["request_id"],
                event["generated_position"],
                event["draft_token_id"],
                event["residual_token_id"],
            )
            if key in seen:
                continue
            seen.add(key)
            request = requests.get(event["request_id"])
            if request is None:
                raise ValueError(f"Missing request record for event {key}")
            position = int(event["generated_position"])
            generated = request["generated_token_ids"]
            if position > len(generated):
                skipped_unreplayable += 1
                print(
                    "Skipping unreplayable terminal-boundary event: "
                    f"position={position}, stored_prefix={len(generated)}, "
                    f"request={event['request_id']}",
                    file=sys.stderr,
                )
                continue
            selected.append(
                {
                    "event": event,
                    "request": request,
                    "generated_prefix": generated[:position],
                }
            )
    if skipped_unreplayable:
        print(
            f"Skipped {skipped_unreplayable} event(s) whose exact prefix was not stored",
            file=sys.stderr,
        )
    return selected


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.max_new_tokens <= 0 or args.batch_size <= 0:
        raise ValueError("max-new-tokens and batch-size must be positive")

    events = load_trace(args.trace_dir, args.event_filter)
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("--sample-ratio must be in (0, 1]")
    total_matched_events = len(events)
    rng = random.Random(args.sample_seed)
    rng.shuffle(events)
    if args.max_events is not None:
        events = events[: args.max_events]
    elif args.sample_ratio < 1:
        sample_size = max(1, round(total_matched_events * args.sample_ratio))
        events = events[:sample_size]
    if not events:
        raise ValueError("No rejection events matched the requested filter")

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    from sglang import Engine

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output_file:
        for begin in range(0, len(events), args.batch_size):
            chunk = events[begin : begin + args.batch_size]
            branch_inputs: list[list[int]] = []
            for item in chunk:
                event = item["event"]
                base = item["request"]["prompt_token_ids"] + item["generated_prefix"]
                branch_inputs.append(base + [event["draft_token_id"]])
                branch_inputs.append(base + [event["residual_token_id"]])

            outputs = engine.generate(
                input_ids=branch_inputs,
                sampling_params=sampling_params,
            )
            if len(outputs) != 2 * len(chunk):
                raise RuntimeError(
                    f"Expected {2 * len(chunk)} branch outputs, got {len(outputs)}"
                )

            for index, item in enumerate(chunk):
                event = item["event"]
                request = item["request"]
                context_ids = request["prompt_token_ids"] + item["generated_prefix"]
                draft_output = outputs[2 * index]
                residual_output = outputs[2 * index + 1]
                record = {
                    "schema_version": 1,
                    "request_id": event["request_id"],
                    "generated_position": event["generated_position"],
                    "context_token_ids": context_ids,
                    "context_text": tokenizer.decode(context_ids),
                    "source_event": event,
                    "draft_branch": {
                        "forced_token_id": event["draft_token_id"],
                        "forced_token_text": tokenizer.decode([event["draft_token_id"]]),
                        "continuation_text": draft_output["text"],
                        "finish_reason": draft_output.get("meta_info", {}).get(
                            "finish_reason"
                        ),
                    },
                    "residual_branch": {
                        "forced_token_id": event["residual_token_id"],
                        "forced_token_text": tokenizer.decode(
                            [event["residual_token_id"]]
                        ),
                        "continuation_text": residual_output["text"],
                        "finish_reason": residual_output.get("meta_info", {}).get(
                            "finish_reason"
                        ),
                    },
                    "replay_config": {
                        "model": args.model,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "seed": args.seed,
                    },
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()

    engine.shutdown()
    print(
        f"Replayed {len(events)} / {total_matched_events} matched rejection events "
        f"(sample_ratio={args.sample_ratio}, sample_seed={args.sample_seed}) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
