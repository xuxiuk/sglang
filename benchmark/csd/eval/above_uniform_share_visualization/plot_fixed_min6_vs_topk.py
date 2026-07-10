#!/usr/bin/env python3
"""Plot fixed above-uniform-share min_count=6 against legacy top-k CSD."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
CURRENT_JSONL = (
    REPO_ROOT
    / "benchmark/csd/runs/lighteval_csd_above_uniform_share/20260605_182153/results/csd_above_uniform_share.jsonl"
)
LEGACY_JSONL = (
    REPO_ROOT
    / "benchmark/csd/runs/lighteval_csd_decoding_comparison_sweep/20260526_014945/results/csd_decoding_comparison_sweep.jsonl"
)
OUT_DIR = Path(__file__).resolve().parent / "figures_fixed_min6_vs_topk"

TASKS = ["LCB", "AIME25", "MATH500"]
METHODS = ["vanilla", "legacy_topk", "ours_min6"]
METHOD_LABELS = {
    "vanilla": "Vanilla EAGLE",
    "legacy_topk": "Top-K CSD\nr=0.3,K=15K",
    "ours_min6": "Ours\n1/K dyn,min=6",
}
COLORS = {
    "vanilla": "#6B7280",
    "legacy_topk": "#7C3AED",
    "ours_min6": "#2563EB",
}


def task_slug(task: str) -> str:
    if task.startswith("lcb:"):
        return "LCB"
    if task.startswith("aime25"):
        return "AIME25"
    if task.startswith("math_500"):
        return "MATH500"
    if task.startswith("gsm8k"):
        return "GSM8K"
    return task.split(":")[0].split("|")[0].upper()


def method_from_run_tag(tag: str, task: str) -> str:
    slug = task_slug(task).lower()
    marker = f"_{slug}_"
    if marker in tag:
        return tag.split(marker, 1)[0]
    for fallback in ["_lcb_", "_aime25_", "_math500_", "_gsm8k_"]:
        if fallback in tag:
            return tag.split(fallback, 1)[0]
    return tag


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_rows() -> pd.DataFrame:
    rows: list[dict] = []

    current = read_jsonl(CURRENT_JSONL)
    legacy = read_jsonl(LEGACY_JSONL)

    current_vanilla: dict[str, dict] = {}
    legacy_vanilla: dict[str, dict] = {}

    for row in current:
        task = task_slug(row["task"])
        tag = row.get("other", {}).get("run_tag", "")
        method = method_from_run_tag(tag, row["task"])
        if method == "vanilla":
            current_vanilla[task] = row

    for row in legacy:
        task = task_slug(row["task"])
        tag = row.get("other", {}).get("run_tag", "")
        method = method_from_run_tag(tag, row["task"])
        if method == "vanilla":
            legacy_vanilla[task] = row

    for task, row in current_vanilla.items():
        rows.append(
            {
                "task": task,
                "method": "vanilla",
                "method_label": METHOD_LABELS["vanilla"],
                "accuracy": row.get("accuracy"),
                "throughput": row.get("throughput"),
                "spec_success_rate": row.get("spec_success_rate"),
                "speedup": 1.0,
                "source": "current",
                "config": "vanilla",
                "run_tag": row.get("other", {}).get("run_tag", ""),
            }
        )

    for row in legacy:
        other = row.get("other", {})
        csd = other.get("csd") or {}
        server = other.get("server_config") or {}
        task = task_slug(row["task"])
        tag = other.get("run_tag", "")
        method = method_from_run_tag(tag, row["task"])
        prob_ratio = csd.get("prob_ratio") or server.get("speculative_csd_prob_ratio")
        top_keep = csd.get("rebuild_top_keep") or server.get(
            "speculative_csd_rebuild_top_keep"
        )
        if method != "csd_plain_table_top5":
            continue
        if prob_ratio != 0.3 or top_keep != 15000.0:
            continue
        base = legacy_vanilla[task]["throughput"]
        rows.append(
            {
                "task": task,
                "method": "legacy_topk",
                "method_label": METHOD_LABELS["legacy_topk"],
                "accuracy": row.get("accuracy"),
                "throughput": row.get("throughput"),
                "spec_success_rate": row.get("spec_success_rate"),
                "speedup": row.get("throughput") / base if base else None,
                "source": "legacy",
                "config": "r=0.3,K=15K",
                "run_tag": tag,
            }
        )

    for row in current:
        other = row.get("other", {})
        csd = other.get("csd") or {}
        task = task_slug(row["task"])
        tag = other.get("run_tag", "")
        method = method_from_run_tag(tag, row["task"])
        if method != "csd_ratio_table":
            continue
        if not csd.get("dynamic_update") or csd.get("dynamic_update_ignore_prob_ratio"):
            continue
        if csd.get("freq_threshold") != 6:
            continue
        base = current_vanilla[task]["throughput"]
        rows.append(
            {
                "task": task,
                "method": "ours_min6",
                "method_label": METHOD_LABELS["ours_min6"],
                "accuracy": row.get("accuracy"),
                "throughput": row.get("throughput"),
                "spec_success_rate": row.get("spec_success_rate"),
                "speedup": row.get("throughput") / base if base else None,
                "source": "current",
                "config": "dynamic_respect,min=6",
                "run_tag": tag,
            }
        )

    df = pd.DataFrame(rows)
    for col in ["accuracy", "throughput", "spec_success_rate", "speedup"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["task"].isin(TASKS)].copy()
    df["task"] = pd.Categorical(df["task"], categories=TASKS, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)
    return df.sort_values(["task", "method"]).reset_index(drop=True)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    *,
    percent: bool = False,
    baseline_line: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8), facecolor="#F8FAFC")
    ax.set_facecolor("#FFFFFF")
    x = np.arange(len(TASKS))
    width = 0.22
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    for off, method in zip(offsets, METHODS):
        sub = df[df["method"] == method].set_index("task").reindex(TASKS)
        vals = sub[metric].to_numpy(dtype=float)
        display_vals = vals * 100 if percent else vals
        bars = ax.bar(
            x + off,
            display_vals,
            width=width * 0.92,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=1.0,
            alpha=0.94,
            zorder=3,
        )
        for bar, val in zip(bars, display_vals):
            if np.isnan(val):
                continue
            if percent:
                text = f"{val:.1f}"
            elif metric == "speedup":
                text = f"{val:.2f}x"
            else:
                text = f"{val:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#111827",
            )

    if baseline_line is not None:
        ax.axhline(
            baseline_line,
            color="#94A3B8",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel(ylabel, color="#111827")
    ax.set_title(title, color="#111827", pad=16, weight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=1.0, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#111827")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = extract_rows()
    df.to_csv(OUT_DIR / "fixed_min6_vs_topk_plot_data.csv", index=False)
    (OUT_DIR / "fixed_min6_vs_topk_plot_data.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2)
    )

    plot_metric(
        df,
        "accuracy",
        "Accuracy (%)",
        "Accuracy: Vanilla vs Top-K CSD vs Ours",
        OUT_DIR / "accuracy_fixed_min6_vs_topk",
        percent=True,
    )
    plot_metric(
        df,
        "speedup",
        "Throughput speedup vs same-run Vanilla (x)",
        "Speedup: Vanilla vs Top-K CSD vs Ours",
        OUT_DIR / "speedup_fixed_min6_vs_topk",
        baseline_line=1.0,
    )
    plot_metric(
        df,
        "throughput",
        "Output throughput (tok/s)",
        "Throughput: Vanilla vs Top-K CSD vs Ours",
        OUT_DIR / "throughput_fixed_min6_vs_topk",
    )
    plot_metric(
        df,
        "spec_success_rate",
        "Speculative success rate (%)",
        "Speculative Success: Vanilla vs Top-K CSD vs Ours",
        OUT_DIR / "spec_success_fixed_min6_vs_topk",
        percent=True,
    )


if __name__ == "__main__":
    main()
