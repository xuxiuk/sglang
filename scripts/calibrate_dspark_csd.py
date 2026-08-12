#!/usr/bin/env python3
"""Run the six-domain RedPajama CSD calibration workload over SGLang HTTP."""

import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
        _thread_local.session.trust_env = False
    return _thread_local.session


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            prompts.append(
                {
                    "index": len(prompts),
                    "domain": row["domain"],
                    "prompt": row["prompt"],
                }
            )
    return prompts


def load_completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip() and not line.startswith("#"):
                completed.add(int(json.loads(line)["index"]))
    return completed


def generate(args, item: dict) -> dict:
    response = _session().post(
        f"http://{args.host}:{args.port}/generate",
        json={
            "text": item["prompt"],
            "sampling_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
            },
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    payload = response.json()
    meta = payload.get("meta_info", {})
    return {
        **item,
        "completion": payload.get("text", ""),
        "completion_tokens": int(meta.get("completion_tokens", 0)),
        "spec_verify_ct": int(meta.get("spec_verify_ct", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    domain_counts = Counter(item["domain"] for item in prompts)
    if len(prompts) != 6000 or set(domain_counts.values()) != {1000}:
        raise RuntimeError(
            f"Expected six domains x 1000 prompts, got {len(prompts)}: {domain_counts}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    completed_indices = load_completed_indices(args.output)
    pending = [item for item in prompts if item["index"] not in completed_indices]
    print(
        f"RedPajama calibration: total={len(prompts)} resumed={len(completed_indices)} "
        f"pending={len(pending)} parallel={args.parallel}",
        flush=True,
    )

    start = time.perf_counter()
    completed_now = 0
    output_tokens = 0
    verify_count = 0
    with args.output.open("a", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(generate, args, item): item for item in pending}
            for future in as_completed(futures):
                row = future.result()
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                completed_now += 1
                output_tokens += row["completion_tokens"]
                verify_count += row["spec_verify_ct"]
                if completed_now % 50 == 0 or completed_now == len(pending):
                    elapsed = time.perf_counter() - start
                    print(
                        f"completed={completed_now}/{len(pending)} "
                        f"output_tok={output_tokens} throughput={output_tokens / elapsed:.2f} tok/s",
                        flush=True,
                    )

    elapsed = time.perf_counter() - start
    summary = {
        "task": "redpajama-csd-calibration",
        "prompt_source": str(args.prompts),
        "num_prompts": len(prompts),
        "domain_counts": dict(domain_counts),
        "resumed_requests": len(completed_indices),
        "completed_requests_this_run": completed_now,
        "parallel": args.parallel,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "elapsed_sec_this_run": elapsed,
        "output_tokens_this_run": output_tokens,
        "output_throughput_this_run": output_tokens / elapsed if elapsed else 0,
        "spec_verify_ct_this_run": verify_count,
        "accept_length_this_run": output_tokens / verify_count if verify_count else None,
    }
    with args.summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
