#!/usr/bin/env python3
"""Generate n samples for local APPS/TACO prompts through DSpark's chat API."""

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("apps", "taco"), required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=81920)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def get_server_info(base_url):
    response = requests.get(f"{base_url.rstrip('/')}/server_info", timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    records = json.loads(args.data_file.read_text(encoding="utf-8"))
    records = records[: args.max_examples]
    jobs = [
        (question_index, sample_index, record)
        for question_index, record in enumerate(records)
        for sample_index in range(args.num_samples)
    ]
    (args.output_dir / "server_info_before.json").write_text(
        json.dumps(get_server_info(args.base_url), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    local = threading.local()

    def run(job):
        question_index, sample_index, record = job
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.trust_env = False
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": record["instruction"]}],
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stream": False,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
        }
        started = time.perf_counter()
        last_error = None
        for attempt in range(3):
            try:
                response = local.session.post(
                    f"{args.base_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    timeout=7200,
                )
                response.raise_for_status()
                body = response.json()
                elapsed = time.perf_counter() - started
                usage = body.get("usage") or {}
                choice = body["choices"][0]
                return {
                    "task": args.task,
                    "question_index": question_index,
                    "sample_index": sample_index,
                    "source_id": record.get("source_id"),
                    "instruction": record["instruction"],
                    "reference": record.get("output"),
                    "output": choice.get("message", {}).get("content", ""),
                    "reasoning_content": choice.get("message", {}).get("reasoning_content"),
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "latency_seconds": elapsed,
                    "attempts": attempt + 1,
                }
            except Exception as exc:
                last_error = repr(exc)
                time.sleep(2**attempt)
        return {
            "task": args.task,
            "question_index": question_index,
            "sample_index": sample_index,
            "source_id": record.get("source_id"),
            "error": last_error,
            "latency_seconds": time.perf_counter() - started,
            "attempts": 3,
        }

    started = time.perf_counter()
    results = []
    output_jsonl = args.output_dir / "generations.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as output, ThreadPoolExecutor(
        max_workers=args.parallel
    ) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            item = future.result()
            results.append(item)
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
            output.flush()
            if completed % 100 == 0 or completed == len(futures):
                print(f"[{args.task}] completed {completed}/{len(futures)}", flush=True)
    wall_seconds = time.perf_counter() - started
    completion_tokens = sum(item.get("completion_tokens", 0) for item in results)
    latencies = [item["latency_seconds"] for item in results if "error" not in item]
    summary = {
        "task": args.task,
        "data_file": str(args.data_file.resolve()),
        "questions": len(records),
        "num_samples": args.num_samples,
        "requests": len(results),
        "successful_requests": len(latencies),
        "failed_requests": len(results) - len(latencies),
        "parallel": args.parallel,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "wall_seconds": wall_seconds,
        "completion_tokens": completion_tokens,
        "aggregate_output_tokens_per_second": completion_tokens / wall_seconds,
        "mean_request_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "median_request_latency_seconds": statistics.median(latencies) if latencies else None,
        "accuracy": None,
        "accuracy_note": "Local records contain no executable tests; pass@4 is not computed.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "server_info_after.json").write_text(
        json.dumps(get_server_info(args.base_url), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
