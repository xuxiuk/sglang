#!/usr/bin/env python3
"""Prepare code-generation OOD datasets for CSD dynamic-update screening."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


DEFAULT_OUT_DIR = Path("benchmark/csd/runs/code_ood_data")


def compact(value: Any, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        text = text[:limit].rstrip() + " ..."
    return text


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return value or "dataset"


def sample_rows(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[i] for i in indices]


def first_solution(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return value
    else:
        parsed = value
    if isinstance(parsed, list) and parsed:
        return compact(parsed[0], 20000)
    return compact(parsed, 20000)


def render_apps(row: dict[str, Any]) -> dict[str, Any]:
    starter = compact(row.get("starter_code"), 4000)
    starter_block = f"\n\nStarter code:\n```python\n{starter}\n```" if starter else ""
    return {
        "instruction": (
            "Solve the competitive programming problem in Python. "
            "Think through the algorithm, then provide the final accepted Python solution. "
            "Return code in a single Python block.\n\n"
            f"Problem:\n{compact(row.get('question'), 18000)}"
            f"{starter_block}\n\nSolution:"
        ),
        "output": first_solution(row.get("solutions")),
        "dataset": "apps",
        "source_id": row.get("id"),
        "domain_key": f"apps_{row.get('difficulty') or 'unknown'}",
    }


def load_apps() -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id="codeparrot/apps",
        filename="test.jsonl",
        repo_type="dataset",
    )
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(render_apps(json.loads(line)))
    return rows


def render_taco(row: dict[str, Any]) -> dict[str, Any]:
    question = row.get("question") or row.get("prompt") or row.get("statement") or ""
    starter = compact(row.get("starter_code"), 4000)
    starter_block = f"\n\nStarter code:\n```python\n{starter}\n```" if starter else ""
    solutions = row.get("solutions") or row.get("solution") or row.get("canonical_solution")
    return {
        "instruction": (
            "Solve the programming contest problem in Python. "
            "Explain the algorithm internally and provide the final accepted Python solution. "
            "Return code in a single Python block.\n\n"
            f"Problem:\n{compact(question, 18000)}"
            f"{starter_block}\n\nSolution:"
        ),
        "output": first_solution(solutions),
        "dataset": "taco",
        "source_id": row.get("id"),
        "domain_key": f"taco_{row.get('difficulty') or 'unknown'}",
    }


def load_taco() -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id="BAAI/TACO",
        filename="ALL/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pd.read_parquet(path)
    return [render_taco(dict(row)) for _, row in table.iterrows()]


def render_lcb(row: dict[str, Any]) -> dict[str, Any]:
    starter = compact(row.get("starter_code"), 4000)
    starter_block = f"\n\nStarter code:\n```python\n{starter}\n```" if starter else ""
    return {
        "instruction": (
            "Solve the LiveCodeBench programming problem in Python. "
            "Think through the algorithm, edge cases, and complexity, then provide the final accepted Python solution. "
            "Return code in a single Python block.\n\n"
            f"Title: {compact(row.get('question_title'))}\n\n"
            f"Problem:\n{compact(row.get('question_content'), 18000)}"
            f"{starter_block}\n\nSolution:"
        ),
        "output": "",
        "dataset": "lcb_codegeneration_v6",
        "source_id": row.get("question_id"),
        "domain_key": f"lcb_{row.get('platform') or 'unknown'}_{row.get('difficulty') or 'unknown'}",
    }


def load_lcb_v6() -> list[dict[str, Any]]:
    dataset = load_dataset("lighteval/code_generation_lite", "v6", split="test")
    return [render_lcb(dict(row)) for row in dataset]


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="apps,taco", help="Comma-separated subset of apps,taco,lcb_v6.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=2000, help="Max rows per dataset; 0 means full.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    loaders = {
        "apps": load_apps,
        "taco": load_taco,
        "lcb_v6": load_lcb_v6,
    }
    written: list[Path] = []
    limit = None if args.limit == 0 else args.limit
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        rows = sample_rows(loaders[name](), limit, args.seed)
        path = args.out_dir / f"{safe_name(name)}_{len(rows)}_code_ood_alpaca.json"
        write_rows(rows, path)
        written.append(path)
        avg_prompt_words = sum(len(row["instruction"].split()) for row in rows) / max(1, len(rows))
        avg_output_words = sum(len(row["output"].split()) for row in rows) / max(1, len(rows))
        max_output_words = max((len(row["output"].split()) for row in rows), default=0)
        print(
            f"{name}: wrote {len(rows)} rows to {path} "
            f"(avg_prompt_words={avg_prompt_words:.1f}, "
            f"avg_output_words={avg_output_words:.1f}, max_output_words={max_output_words})"
        )
    print('OOD_DATA_FILES="' + " ".join(str(path) for path in written) + '"')


if __name__ == "__main__":
    main()
