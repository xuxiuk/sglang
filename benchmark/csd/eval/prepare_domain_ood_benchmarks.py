#!/usr/bin/env python3
"""Prepare domain OOD benchmark JSON files for CSD dynamic-update screening.

The generated files use the AlpacaEval-compatible shape consumed by
bench_sglang_chat_generation.py with --dataset alpaca_eval --data-file.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


DEFAULT_OUT_DIR = Path("benchmark/csd/runs/domain_ood_data")

LEGALBENCH_TASKS = [
    "cuad_governing_law",
    "cuad_affiliate_license-licensee",
    "cuad_audit_rights",
    "cuad_cap_on_liability",
    "cuad_change_of_control",
    "cuad_effective_date",
    "cuad_expiration_date",
    "cuad_insurance",
    "cuad_license_grant",
    "cuad_non-compete",
    "cuad_termination_for_convenience",
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_sharing_with_third-parties",
    "contract_nli_survival_of_obligations",
    "contract_qa",
    "consumer_contracts_qa",
]

OPTION_LABELS = ["A", "B", "C", "D"]


def compact(text: Any, limit: int | None = None) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"\s+", " ", value).strip()
    if limit is not None and len(value) > limit:
        value = value[:limit].rstrip() + " ..."
    return value


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return value or "dataset"


def maybe_shuffle(rows: list[dict[str, Any]], order: str, seed: int) -> list[dict[str, Any]]:
    rows = list(rows)
    if order == "shuffled":
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows


def truncate_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[:limit]


def balanced_grouped(
    buckets: list[list[dict[str, Any]]], limit: int | None
) -> list[dict[str, Any]]:
    """Keep groups contiguous while preventing large early groups from owning the limit."""
    non_empty = [bucket for bucket in buckets if bucket]
    if limit is None:
        return [item for bucket in non_empty for item in bucket]
    if not non_empty or limit <= 0:
        return []
    quota = max(1, limit // len(non_empty))
    selected: list[list[dict[str, Any]]] = []
    remaining: list[list[dict[str, Any]]] = []
    used = 0
    for bucket in non_empty:
        take = min(len(bucket), quota)
        selected.append(bucket[:take])
        used += take
        if take < len(bucket):
            remaining.append(bucket[take:])
    cursor = 0
    while used < limit and remaining:
        bucket = remaining[cursor % len(remaining)]
        if bucket:
            selected[cursor % len(selected)].append(bucket.pop(0))
            used += 1
        remaining = [bucket for bucket in remaining if bucket]
        cursor += 1
    return [item for bucket in selected for item in bucket][:limit]


def legalbench_clause_name(config: str) -> str:
    if config.startswith("cuad_"):
        name = config[len("cuad_") :]
        return name.replace("-", " ").replace("_", " ")
    if config.startswith("contract_nli_"):
        name = config[len("contract_nli_") :]
        return name.replace("-", " ").replace("_", " ")
    return config.replace("-", " ").replace("_", " ")


def render_legalbench(config: str, row: dict[str, Any]) -> dict[str, Any]:
    answer = compact(row.get("answer", ""))
    if config == "consumer_contracts_qa":
        instruction = (
            "Read the consumer contract excerpt and answer the legal question. "
            "Give a concise reason and end with `FINAL: Yes` or `FINAL: No`.\n\n"
            f"Contract excerpt:\n{compact(row.get('contract'), 6000)}\n\n"
            f"Question:\n{compact(row.get('question'))}\n\nAnswer:"
        )
    elif config == "contract_qa":
        instruction = (
            "Read the contract clause and answer the question. Give one concise legal reason "
            "and end with `FINAL: Yes` or `FINAL: No`.\n\n"
            f"Clause:\n{compact(row.get('text'), 6000)}\n\n"
            f"Question:\n{compact(row.get('question'))}\n\nAnswer:"
        )
    elif config.startswith("cuad_"):
        clause = legalbench_clause_name(config)
        instruction = (
            "Review the contract clause for the requested CUAD issue. Answer with a concise reason "
            "and end with `FINAL: Yes` or `FINAL: No`.\n\n"
            f"Requested issue: {clause}\n\n"
            f"Clause:\n{compact(row.get('text'), 6000)}\n\nAnswer:"
        )
    elif config.startswith("contract_nli_"):
        issue = legalbench_clause_name(config)
        instruction = (
            "Determine whether the contract text entails the requested legal proposition. "
            "Give one concise reason and end with `FINAL: Yes` or `FINAL: No`.\n\n"
            f"Proposition: {issue}\n\n"
            f"Contract text:\n{compact(row.get('text'), 6000)}\n\nAnswer:"
        )
    else:
        instruction = (
            "Read the legal text and answer the task. Give one concise reason and end with `FINAL:`.\n\n"
            f"Task: {legalbench_clause_name(config)}\n\n"
            f"Text:\n{compact(row.get('text') or row.get('contract'), 6000)}\n\nAnswer:"
        )
    return {
        "instruction": instruction,
        "output": f"The answer is {answer}. FINAL: {answer}",
        "dataset": f"legalbench_{config}",
        "source_id": row.get("index"),
        "domain_key": config,
    }


def prepare_legalbench(limit: int | None, order: str, seed: int) -> list[dict[str, Any]]:
    buckets: list[list[dict[str, Any]]] = []
    for config in LEGALBENCH_TASKS:
        ds = load_dataset("nguha/legalbench", config, split="test")
        buckets.append([render_legalbench(config, dict(row)) for row in ds])
    if order == "grouped":
        return balanced_grouped(buckets, limit)
    rows = [item for bucket in buckets for item in bucket]
    rows = maybe_shuffle(rows, order, seed)
    return truncate_rows(rows, limit)


def render_medmcqa(row: dict[str, Any]) -> dict[str, Any]:
    options = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
    cop = int(row.get("cop", 0))
    label = OPTION_LABELS[cop] if 0 <= cop < len(OPTION_LABELS) else "A"
    option_text = compact(options[cop]) if 0 <= cop < len(options) else ""
    subject = compact(row.get("subject_name") or "Medicine")
    topic = compact(row.get("topic_name") or "")
    exp = compact(row.get("exp") or option_text, 500)
    topic_line = f"\nTopic: {topic}" if topic else ""
    instruction = (
        "Answer the medical multiple-choice question. Give the best option and one concise medical rationale. "
        "End with `FINAL: <option letter>`.\n\n"
        f"Subject: {subject}{topic_line}\n\n"
        f"Question: {compact(row.get('question'))}\n"
        f"A. {compact(options[0])}\n"
        f"B. {compact(options[1])}\n"
        f"C. {compact(options[2])}\n"
        f"D. {compact(options[3])}\n\nAnswer:"
    )
    return {
        "instruction": instruction,
        "output": f"{label}. {option_text}. {exp} FINAL: {label}",
        "dataset": "medmcqa",
        "source_id": row.get("id"),
        "domain_key": subject,
    }


def prepare_medmcqa(limit: int | None, order: str, seed: int) -> list[dict[str, Any]]:
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    rows = [render_medmcqa(dict(row)) for row in ds]
    if order == "grouped":
        buckets_by_subject: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            buckets_by_subject.setdefault(str(row.get("domain_key") or ""), []).append(row)
        buckets = [buckets_by_subject[key] for key in sorted(buckets_by_subject)]
        return balanced_grouped(buckets, limit)
    rows = maybe_shuffle(rows, order, seed)
    return truncate_rows(rows, limit)


def render_finqa(row: dict[str, Any]) -> dict[str, Any]:
    query = compact(row.get("query"), 7000)
    answer = compact(row.get("answer"))
    question = compact(row.get("text"))
    instruction = (
        "Answer the financial question using the provided context. Show a brief calculation or reasoning step, "
        "then end with `FINAL: <answer>`.\n\n"
        f"{query}\n\n"
        f"Question summary: {question}\n\nAnswer:"
    )
    return {
        "instruction": instruction,
        "output": f"Using the provided financial figures, the answer is {answer}. FINAL: {answer}",
        "dataset": "finqa",
        "source_id": row.get("id"),
        "domain_key": "finqa",
    }


def prepare_finqa(limit: int | None, order: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    for split in ("valid", "test", "train"):
        ds = load_dataset("ChanceFocus/flare-finqa", split=split)
        rows.extend(render_finqa(dict(row)) for row in ds)
        if limit is not None and len(rows) >= limit and order == "grouped":
            break
    rows = maybe_shuffle(rows, order, seed)
    return truncate_rows(rows, limit)


def pubmed_context(context: Any) -> str:
    if isinstance(context, dict):
        contexts = context.get("contexts", [])
        labels = context.get("labels", [])
    else:
        try:
            parsed = ast.literal_eval(str(context))
            contexts = parsed.get("contexts", []) if isinstance(parsed, dict) else []
            labels = parsed.get("labels", []) if isinstance(parsed, dict) else []
        except Exception:
            contexts = [str(context)]
            labels = []
    parts = []
    for idx, sentence in enumerate(contexts):
        label = labels[idx] if idx < len(labels) else ""
        prefix = f"[{label}] " if label else ""
        parts.append(prefix + compact(sentence))
    return "\n".join(parts)


def render_pubmedqa(row: dict[str, Any]) -> dict[str, Any]:
    decision = compact(row.get("final_decision"))
    long_answer = compact(row.get("long_answer"), 700)
    instruction = (
        "Answer the biomedical research question from the PubMed abstract snippets. "
        "Use yes, no, or maybe, give one concise evidence-based rationale, and end with `FINAL: yes/no/maybe`.\n\n"
        f"Question: {compact(row.get('question'))}\n\n"
        f"Abstract snippets:\n{compact(pubmed_context(row.get('context')), 7000)}\n\nAnswer:"
    )
    return {
        "instruction": instruction,
        "output": f"{decision}. {long_answer} FINAL: {decision}",
        "dataset": "pubmedqa_labeled",
        "source_id": row.get("pubid"),
        "domain_key": "pubmedqa",
    }


def prepare_pubmedqa(limit: int | None, order: str, seed: int) -> list[dict[str, Any]]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = [render_pubmedqa(dict(row)) for row in ds]
    rows = maybe_shuffle(rows, order, seed)
    return truncate_rows(rows, limit)


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset(name: str, limit: int | None, order: str, seed: int) -> list[dict[str, Any]]:
    if name == "legalbench_cuad":
        return prepare_legalbench(limit, order, seed)
    if name == "medmcqa":
        return prepare_medmcqa(limit, order, seed)
    if name == "finqa":
        return prepare_finqa(limit, order, seed)
    if name == "pubmedqa":
        return prepare_pubmedqa(limit, order, seed)
    raise ValueError(f"Unknown dataset: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="legalbench_cuad,finqa,medmcqa,pubmedqa",
        help="Comma-separated subset of legalbench_cuad, finqa, medmcqa, pubmedqa.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order", choices=["grouped", "shuffled"], default="grouped")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    written = []
    for name in names:
        rows = build_dataset(name, args.limit, args.order, args.seed)
        filename = f"{safe_name(name)}_{args.order}_{len(rows)}_alpaca.json"
        path = args.out_dir / filename
        write_rows(rows, path)
        written.append(path)
        print(f"{name}: wrote {len(rows)} rows to {path}")
    print("OOD_DATA_FILES=\"" + " ".join(str(path) for path in written) + "\"")


if __name__ == "__main__":
    main()
