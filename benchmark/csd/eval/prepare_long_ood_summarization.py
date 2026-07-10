#!/usr/bin/env python3
"""Prepare long-output OOD summarization datasets for CSD dynamic-update tests."""

from __future__ import annotations

import argparse
import json
import re
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_OUT_DIR = Path("benchmark/csd/runs/long_ood_data")


def compact(value: Any, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        text = text[:limit].rstrip() + " ..."
    return text


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return value or "dataset"


def split_prefix(split: str, limit: int) -> str:
    return f"{split}[:{limit}]"


def take(iterable: Any, limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return [dict(row) for row in iterable]
    return [dict(row) for row in islice(iterable, limit)]


def render_pubmed(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": (
            "Write a detailed biomedical abstract-style summary of the PubMed article. "
            "Preserve the main objective, methods, findings, and conclusion.\n\n"
            f"Article:\n{compact(row.get('article'), 18000)}\n\nSummary:"
        ),
        "output": compact(row.get("abstract"), 6000),
        "dataset": "pubmed_summarization",
        "domain_key": "biomedical_summarization",
    }


def render_arxiv(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": (
            "Write a detailed scientific abstract-style summary of the arXiv paper. "
            "Preserve the research problem, method, experiments, and conclusions.\n\n"
            f"Paper:\n{compact(row.get('article'), 18000)}\n\nSummary:"
        ),
        "output": compact(row.get("abstract"), 6000),
        "dataset": "arxiv_summarization",
        "domain_key": "scientific_summarization",
    }


def render_govreport(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": (
            "Write a detailed government report summary. Cover the main findings, agencies, "
            "policy implications, and recommendations.\n\n"
            f"Report:\n{compact(row.get('report'), 18000)}\n\nSummary:"
        ),
        "output": compact(row.get("summary"), 6000),
        "dataset": "govreport_summarization",
        "domain_key": "government_report_summarization",
    }


def render_cnn_dailymail(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": (
            "Write a detailed news summary of the article. Preserve the main events, actors, "
            "background, and consequences.\n\n"
            f"Article:\n{compact(row.get('article'), 18000)}\n\nSummary:"
        ),
        "output": compact(row.get("highlights"), 6000),
        "dataset": "cnn_dailymail",
        "domain_key": "news_summarization",
    }


def render_multi_news(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": (
            "Write a detailed multi-document news summary. Synthesize the shared facts and "
            "important differences across the source documents.\n\n"
            f"Documents:\n{compact(row.get('document'), 18000)}\n\nSummary:"
        ),
        "output": compact(row.get("summary"), 6000),
        "dataset": "multi_news",
        "domain_key": "multi_document_news_summarization",
    }


def render_billsum(row: dict[str, Any]) -> dict[str, Any]:
    title = compact(row.get("title"))
    title_line = f"Title: {title}\n\n" if title else ""
    return {
        "instruction": (
            "Write a detailed legislative summary of the bill. Cover the main policy changes, "
            "affected parties, and important provisions.\n\n"
            f"{title_line}Bill text:\n{compact(row.get('text'), 18000)}\n\nSummary:"
        ),
        "output": compact(row.get("summary"), 6000),
        "dataset": "billsum",
        "domain_key": "legislative_summarization",
    }


def render_flare_edtsum(row: dict[str, Any]) -> dict[str, Any]:
    query = compact(row.get("query"))
    query_line = f"Task: {query}\n\n" if query else ""
    return {
        "instruction": (
            "Write a detailed financial disclosure summary. Include the important financial events, "
            "business drivers, and quantitative details when present.\n\n"
            f"{query_line}Financial document:\n{compact(row.get('text'), 18000)}\n\nSummary:"
        ),
        "output": compact(row.get("answer"), 6000),
        "dataset": "flare_edtsum",
        "source_id": row.get("id"),
        "domain_key": "financial_summarization",
    }


def render_legal_case(row: dict[str, Any]) -> dict[str, Any]:
    dataset_name = compact(row.get("dataset_name"))
    dataset_line = f"Source: {dataset_name}\n\n" if dataset_name else ""
    return {
        "instruction": (
            "Write a detailed legal case summary. Cover the key facts, legal issues, reasoning, "
            "and final disposition.\n\n"
            f"{dataset_line}Judgement:\n{compact(row.get('judgement'), 18000)}\n\nSummary:"
        ),
        "output": compact(row.get("summary"), 6000),
        "dataset": "legal_case_document_summarization",
        "domain_key": "legal_case_summarization",
    }


def render_lighteval_legal(row: dict[str, Any], subset: str) -> dict[str, Any]:
    subset_label = {
        "MultiLexSum": "legal case multi-document",
        "EurLexSum": "European Union legal act",
        "BillSum": "legislative bill",
    }[subset]
    return {
        "instruction": (
            f"Write a detailed {subset_label} summary. Cover the key facts, legal issues, "
            "important provisions, reasoning, and final outcome when available.\n\n"
            f"Document:\n{compact(row.get('article'), 18000)}\n\nSummary:"
        ),
        "output": compact(row.get("summary"), 6000),
        "dataset": f"lighteval_legal_summarization_{subset}",
        "domain_key": f"legal_summarization_{subset.lower()}",
    }


def load_rows(name: str, limit: int | None, seed: int) -> list[dict[str, Any]]:
    del seed
    if name == "pubmed_summarization":
        ds = load_dataset(
            "ccdv/pubmed-summarization",
            "document",
            split="validation",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_pubmed(row) for row in take(ds, limit)]
    if name == "arxiv_summarization":
        ds = load_dataset(
            "ccdv/arxiv-summarization",
            "document",
            split="validation",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_arxiv(row) for row in take(ds, limit)]
    if name == "govreport_summarization":
        ds = load_dataset(
            "ccdv/govreport-summarization",
            "document",
            split="validation",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_govreport(row) for row in take(ds, limit)]
    if name == "billsum":
        ds = load_dataset(
            "billsum",
            split="test",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_billsum(row) for row in take(ds, limit)]
    if name == "cnn_dailymail":
        ds = load_dataset(
            "abisee/cnn_dailymail",
            "3.0.0",
            split="test",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_cnn_dailymail(row) for row in take(ds, limit)]
    if name == "multi_news":
        ds = load_dataset(
            "multi_news",
            split="test",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_multi_news(row) for row in take(ds, limit)]
    if name == "flare_edtsum":
        ds = load_dataset(
            "ChanceFocus/flare-edtsum",
            split="test",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_flare_edtsum(row) for row in take(ds, limit)]
    if name == "legal_case_summary":
        ds = load_dataset(
            "joelniklaus/legal_case_document_summarization",
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_legal_case(row) for row in take(ds, limit)]
    if name == "lighteval_legal_multilexsum":
        ds = load_dataset(
            "lighteval/legal_summarization",
            "MultiLexSum",
            split="test",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_lighteval_legal(row, "MultiLexSum") for row in take(ds, limit)]
    if name == "lighteval_legal_eurlexsum":
        ds = load_dataset(
            "lighteval/legal_summarization",
            "EurLexSum",
            split="test",
            streaming=True,
            trust_remote_code=False,
        )
        return [render_lighteval_legal(row, "EurLexSum") for row in take(ds, limit)]
    raise ValueError(f"Unknown dataset: {name}")


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="pubmed_summarization,billsum,flare_edtsum",
        help=(
            "Comma-separated subset of pubmed_summarization,billsum,flare_edtsum,"
            "legal_case_summary,cnn_dailymail,multi_news,arxiv_summarization,"
            "govreport_summarization,lighteval_legal_multilexsum,"
            "lighteval_legal_eurlexsum."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=1000, help="Number of examples per dataset; 0 means full split.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    written: list[Path] = []
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        limit = None if args.limit == 0 else args.limit
        rows = load_rows(name, limit, args.seed)
        path = args.out_dir / f"{safe_name(name)}_{len(rows)}_long_alpaca.json"
        write_rows(rows, path)
        written.append(path)
        avg_out_words = sum(len(row["output"].split()) for row in rows) / max(1, len(rows))
        avg_prompt_words = sum(len(row["instruction"].split()) for row in rows) / max(1, len(rows))
        print(
            f"{name}: wrote {len(rows)} rows to {path} "
            f"(avg_prompt_words={avg_prompt_words:.1f}, avg_output_words={avg_out_words:.1f})"
        )
    print('OOD_DATA_FILES="' + " ".join(str(path) for path in written) + '"')


if __name__ == "__main__":
    main()
