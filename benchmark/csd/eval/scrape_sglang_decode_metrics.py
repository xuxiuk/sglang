#!/usr/bin/env python3
"""Scrape SGLang decode-token metrics at a fixed interval."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def metric_sum(text: str, name: str, required_labels: dict[str, str] | None = None) -> float | None:
    required_labels = required_labels or {}
    total = 0.0
    found = False
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{(?P<labels>[^}}]*)\}})?\s+(?P<value>[-+eE0-9.]+)$")
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        labels = match.group("labels") or ""
        if any(f'{key}="{value}"' not in labels for key, value in required_labels.items()):
            continue
        total += float(match.group("value"))
        found = True
    return total if found else None


def fetch_metrics(url: str, timeout: float) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/metrics"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prev_time = None
    prev_decode_tokens = None

    with args.output.open("w", encoding="utf-8") as fout:
        while True:
            now = time.time()
            try:
                text = fetch_metrics(url, args.timeout)
            except (OSError, URLError):
                break

            decode_tokens = metric_sum(
                text,
                "sglang:realtime_tokens_total",
                required_labels={"mode": "decode"},
            )
            prefill_compute_tokens = metric_sum(
                text,
                "sglang:realtime_tokens_total",
                required_labels={"mode": "prefill_compute"},
            )
            gen_throughput = metric_sum(text, "sglang:gen_throughput")
            running_reqs = metric_sum(text, "sglang:num_running_reqs")
            queue_reqs = metric_sum(text, "sglang:num_queue_reqs")

            delta_decode_tokens = None
            delta_seconds = None
            decode_tok_s = None
            if (
                decode_tokens is not None
                and prev_decode_tokens is not None
                and prev_time is not None
            ):
                delta_decode_tokens = decode_tokens - prev_decode_tokens
                delta_seconds = now - prev_time
                if delta_seconds > 0:
                    decode_tok_s = delta_decode_tokens / delta_seconds

            record = {
                "timestamp": now,
                "decode_tokens_total": decode_tokens,
                "prefill_compute_tokens_total": prefill_compute_tokens,
                "delta_decode_tokens": delta_decode_tokens,
                "delta_seconds": delta_seconds,
                "decode_tok_s": decode_tok_s,
                "gen_throughput_gauge": gen_throughput,
                "running_reqs": running_reqs,
                "queue_reqs": queue_reqs,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            prev_time = now
            prev_decode_tokens = decode_tokens
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
