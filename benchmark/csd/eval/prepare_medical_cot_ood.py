#!/usr/bin/env python3
"""Prepare medical CoT OOD datasets for CSD dynamic-update screening."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


DEFAULT_OUT_DIR = Path("benchmark/csd/runs/medical_cot_ood_data")


DATASETS = {
    "medical_o1": (
        "FreedomIntelligence/medical-o1-reasoning-SFT",
        "medical_o1_sft.json",
    ),
    "medprompt_medqa_cot": (
        "HPAI-BSC/Medprompt-MedQA-CoT",
        "medprompt_medqa_cot_llama31.json",
    ),
    "reasonmed": (
        "lingshu-medical-mllm/ReasonMed",
        "ReasonMed.json",
    ),
}


def compact(value: Any, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        text = text[:limit].rstrip() + " ..."
    return text


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return value or "dataset"


def load_json(repo_id: str, filename: str) -> Any:
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return json.load(open(path, "r", encoding="utf-8"))


def sample_rows(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[i] for i in indices]


def render_medical_o1(row: dict[str, Any]) -> dict[str, Any]:
    question = compact(row.get("Question"), 8000)
    cot = compact(row.get("Complex_CoT"), 12000)
    response = compact(row.get("Response"), 4000)
    return {
        "instruction": (
            "Solve the medical question with detailed step-by-step clinical reasoning. "
            "Discuss the relevant findings, differential diagnosis, and why the final answer follows. "
            "End with a concise final answer.\n\n"
            f"Question:\n{question}\n\nAnswer:"
        ),
        "output": f"<think>\n{cot}\n</think>\n\n{response}",
        "dataset": "medical_o1_reasoning_sft",
        "domain_key": "medical_o1_cot",
    }


def medprompt_options(options: Any) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{k}. {compact(v)}" for k, v in sorted(options.items()))
    return compact(options)


def render_medprompt(row: dict[str, Any]) -> dict[str, Any]:
    generations = row.get("generations") or []
    first_generation = generations[0] if generations and isinstance(generations[0], dict) else {}
    rationale = compact(first_generation.get("response"), 12000)
    answer = compact(row.get("correct_answer"))
    return {
        "instruction": (
            "Answer the USMLE-style medical multiple-choice question with detailed step-by-step reasoning. "
            "Analyze the clinical presentation and options before giving the final option. "
            "End with `FINAL: <option letter>`.\n\n"
            f"Question:\n{compact(row.get('question'), 8000)}\n\n"
            f"Options:\n{medprompt_options(row.get('options'))}\n\nAnswer:"
        ),
        "output": f"{rationale}\n\nFINAL: {answer}",
        "dataset": "medprompt_medqa_cot",
        "source_id": row.get("_id") or row.get("id"),
        "domain_key": "medprompt_medqa_cot",
    }


def render_reasonmed(row: dict[str, Any]) -> dict[str, Any]:
    instruction = compact(row.get("instruction"), 10000)
    extra_input = compact(row.get("input"), 8000)
    input_block = f"\n\nAdditional context:\n{extra_input}" if extra_input else ""
    return {
        "instruction": (
            "Solve the following medical reasoning problem. Think step by step in detail, "
            "then provide the final answer.\n\n"
            f"{instruction}{input_block}\n\nAnswer:"
        ),
        "output": compact(row.get("output"), 20000),
        "dataset": "reasonmed",
        "domain_key": "reasonmed_cot",
    }


def normalize_dataset(name: str, data: Any) -> list[dict[str, Any]]:
    if name == "medical_o1":
        return [render_medical_o1(dict(row)) for row in data]
    if name == "medprompt_medqa_cot":
        rows = []
        for key, value in data.items():
            row = dict(value)
            row["_id"] = key
            rows.append(render_medprompt(row))
        return rows
    if name == "reasonmed":
        return [render_reasonmed(dict(row)) for row in data]
    raise ValueError(f"Unknown dataset: {name}")


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="medical_o1,medprompt_medqa_cot,reasonmed",
        help="Comma-separated subset of medical_o1, medprompt_medqa_cot, reasonmed.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=2000, help="Max rows per dataset; 0 means full.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    written: list[Path] = []
    limit = None if args.limit == 0 else args.limit
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        repo_id, filename = DATASETS[name]
        rows = normalize_dataset(name, load_json(repo_id, filename))
        rows = sample_rows(rows, limit, args.seed)
        path = args.out_dir / f"{safe_name(name)}_{len(rows)}_medical_cot_alpaca.json"
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
