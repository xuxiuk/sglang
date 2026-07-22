#!/usr/bin/env python3
"""Summarize CSD domain-OOD benchmark result JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

METHOD_ORDER = [
    "auto",
    "eagle",
    "plain",
    "dynamic",
    "dynamic_ignore_ratio",
    "dynamic_ignore_ratio_huge",
    "dynamic_entropy_p20_ignore_ratio",
    "dynamic_entropy_p20_ignore_ratio_huge",
]


def infer_method(row: dict[str, Any]) -> str:
    haystacks = [
        str(row.get("model_id") or ""),
        str(row.get("other", {}).get("model_id") or ""),
        str(row.get("other", {}).get("answer_file") or ""),
    ]
    for method in [
        "dynamic_entropy_p20_ignore_ratio_huge",
        "dynamic_entropy_p20_ignore_ratio",
        "dynamic_ignore_ratio_huge",
        "dynamic_ignore_ratio",
        "dynamic",
        "plain",
        "eagle",
        "auto",
    ]:
        if any(method in value for value in haystacks):
            return method
    return "unknown"


def infer_dataset(row: dict[str, Any], fallback: str) -> str:
    task = row.get("task")
    if task and task != "alpaca_eval":
        return str(task)
    other = row.get("other", {})
    data_file = other.get("data_file")
    if data_file:
        name = Path(data_file).name
        for suffix in ["_alpaca.json", ".jsonl", ".json"]:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name
    answer_file = other.get("answer_file")
    if answer_file:
        name = Path(answer_file).name
        for method in METHOD_ORDER:
            name = name.replace(method + "_", "")
        return name.replace("_model_outputs.json", "").replace("_answers.jsonl", "")
    return fallback


def load_rows(paths: list[Path]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                rows.append((path.stem, json.loads(line)))
    return rows


def pct(value: float) -> str:
    return f"{value:+.2f}%"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--positive-threshold", type=float, default=3.0)
    parser.add_argument("--draft-tokens", type=float, default=5.0)
    args = parser.parse_args()

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for fallback, row in load_rows(args.results):
        method = infer_method(row)
        dataset = infer_dataset(row, fallback)
        grouped[dataset][method] = row

    positives = []
    for dataset in sorted(grouped):
        methods = grouped[dataset]
        plain = methods.get("plain")
        print(f"\n## {dataset}")
        print(
            "| method | tok/s | head90 tok/s | win30 p90 tok/s | win30 p90 tok | win30 max tok | win60 p90 tok/s | Δtok/s vs plain | accept | spec% | avg out | p90 out | max out | hit | hit% | tail tok% | tail time% | avg lat | p90 lat | p90 req tok/s | avg prompt |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for method in METHOD_ORDER + [m for m in sorted(methods) if m not in METHOD_ORDER]:
            row = methods.get(method)
            if row is None:
                continue
            tok = float(row.get("throughput") or 0.0)
            plain_tok = float(plain.get("throughput") or 0.0) if plain else 0.0
            delta = (tok / plain_tok - 1.0) * 100.0 if plain_tok > 0 else 0.0
            accept = float(row.get("accept_length") or 1.0)
            spec = max(0.0, (accept - 1.0) / args.draft_tokens * 100.0)
            hits = int(row.get("max_new_token_hits") or 0)
            n = int(row.get("num_requests") or 0)
            hit_rate = hits / n * 100.0 if n else 0.0
            tail_token_share = float(row.get("tail10_token_share") or 0.0) * 100.0
            tail_time_share = float(row.get("tail10_time_share") or 0.0) * 100.0
            print(
                f"| `{method}` | {tok:.1f} | {fmt(row.get('head90_throughput'), 1)} | "
                f"{fmt(row.get('p90_completed_30s_throughput'), 1)} | "
                f"{fmt(row.get('p90_completed_30s_tokens'), 0)} | "
                f"{fmt(row.get('max_completed_30s_tokens'), 0)} | "
                f"{fmt(row.get('p90_completed_60s_throughput'), 1)} | "
                f"{pct(delta) if plain else '-'} | "
                f"{accept:.3f} | {spec:.2f} | {fmt(row.get('avg_completion_tokens'), 1)} | "
                f"{fmt(row.get('p90_completion_tokens'), 0)} | "
                f"{fmt(row.get('max_completion_tokens'), 0)} | {hits}/{n} | {hit_rate:.2f}% | "
                f"{tail_token_share:.1f}% | {tail_time_share:.1f}% | "
                f"{fmt(row.get('avg_request_latency'), 1)} | {fmt(row.get('p90_request_latency'), 1)} | "
                f"{fmt(row.get('p90_request_throughput'), 1)} | "
                f"{fmt(row.get('avg_prompt_tokens'), 1)} |"
            )
            if plain and method.startswith("dynamic"):
                avg = float(row.get("avg_completion_tokens") or 0.0)
                plain_avg = float(plain.get("avg_completion_tokens") or 0.0)
                avg_ratio = avg / plain_avg if plain_avg > 0 else 1.0
                plain_hits = int(plain.get("max_new_token_hits") or 0)
                plain_n = int(plain.get("num_requests") or 0)
                plain_hit_rate = plain_hits / plain_n if plain_n else 0.0
                suspicious_short = avg_ratio < 0.95
                worse_hits = hit_rate / 100.0 > plain_hit_rate + 0.02
                if delta >= args.positive_threshold and not suspicious_short and not worse_hits:
                    positives.append((dataset, method, delta, accept - float(plain.get("accept_length") or 1.0)))

    if positives:
        print("\n## Candidate positives")
        for dataset, method, delta, accept_delta in positives:
            print(
                f"- {dataset}: `{method}` is {delta:.2f}% faster than `plain`, "
                f"accept length Δ={accept_delta:+.3f}."
            )
    else:
        print("\n## Candidate positives")
        print("- None passed the configured positive threshold.")


if __name__ == "__main__":
    main()
