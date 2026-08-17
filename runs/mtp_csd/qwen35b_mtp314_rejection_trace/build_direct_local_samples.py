#!/usr/bin/env python3
"""Build a globally sampled direct-local-judge dataset from rejection traces."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tokenizer", default="/data/model/Qwen3.5-35B-A3B")
    p.add_argument("--sample-ratio", type=float, default=0.05)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--max-events", type=int)
    return p.parse_args()


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    args = parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite output or manifest")
    if not 0 < args.sample_ratio <= 1:
        raise ValueError("--sample-ratio must be in (0, 1]")

    events = []
    seen = set()
    raw_by_task = {}
    for task in args.tasks:
        trace_dir = args.trace_root / task / "trace"
        raw = 0
        for path in sorted(trace_dir.glob("events.rank*.jsonl")):
            for event in read_jsonl(path):
                raw += 1
                key = (
                    task,
                    event["request_id"],
                    int(event["generated_position"]),
                    int(event["draft_token_id"]),
                    int(event["residual_token_id"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                events.append((task, event))
        raw_by_task[task] = raw

    rng = random.Random(args.sample_seed)
    rng.shuffle(events)
    total_unique = len(events)
    sample_size = min(total_unique, max(1, round(total_unique * args.sample_ratio)))
    if args.max_events is not None:
        sample_size = min(sample_size, args.max_events)
    selected = events[:sample_size]
    needed = {task: set() for task in args.tasks}
    for task, event in selected:
        needed[task].add(event["request_id"])

    requests = {task: {} for task in args.tasks}
    for task in args.tasks:
        trace_dir = args.trace_root / task / "trace"
        for path in sorted(trace_dir.glob("requests.rank*.jsonl")):
            for request in read_jsonl(path):
                if request["request_id"] in needed[task]:
                    requests[task][request["request_id"]] = request

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_terminal = 0
    by_task = {task: 0 for task in args.tasks}
    with args.output.open("x", encoding="utf-8", buffering=1) as handle:
        for sample_index, (task, event) in enumerate(selected):
            request = requests[task].get(event["request_id"])
            if request is None:
                raise ValueError(f"Missing request {task}:{event['request_id']}")
            position = int(event["generated_position"])
            generated = request["generated_token_ids"]
            if position > len(generated):
                skipped_terminal += 1
                continue
            context_ids = request["prompt_token_ids"] + generated[:position]
            draft_id = int(event["draft_token_id"])
            residual_id = int(event["residual_token_id"])
            record = {
                "schema_version": 1,
                "sample_index": written,
                "sample_id": (
                    f"{task}:{event['request_id']}:{position}:{draft_id}:{residual_id}"
                ),
                "task_name": task,
                "request_id": event["request_id"],
                "generated_position": position,
                "context_text": tokenizer.decode(context_ids),
                "draft_token_id": draft_id,
                "draft_token_text": tokenizer.decode([draft_id]),
                "residual_token_id": residual_id,
                "residual_token_text": tokenizer.decode([residual_id]),
                "source_event": event,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            by_task[task] += 1

    manifest = {
        "schema_version": 1,
        "trace_root": str(args.trace_root.resolve()),
        "tasks": args.tasks,
        "raw_events_by_task": raw_by_task,
        "unique_events_before_sampling": total_unique,
        "sampling": {
            "strategy": "global_uniform_without_replacement",
            "sample_ratio": args.sample_ratio,
            "sample_seed": args.sample_seed,
            "requested_sample_size": sample_size,
        },
        "written_events": written,
        "written_events_by_task": by_task,
        "skipped_unreplayable_terminal_events": skipped_terminal,
        "tokenizer": args.tokenizer,
        "context_definition": "prompt_token_ids + generated_token_ids[:generated_position]",
        "candidate_suffix_tokens": 0,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if skipped_terminal:
        print(f"Skipped {skipped_terminal} terminal-boundary event(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
