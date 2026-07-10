#!/usr/bin/env python3
"""Plot LightEval CSD decoding comparison results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ["LCB", "AIME25", "MATH500", "GSM8K"]
METHOD_ORDER = [
    "vanilla",
    "csd_plain_table_static",
    "csd_plain_table",
    "csd_plain_table_top5",
]
METHOD_LABELS = {
    "vanilla": "Vanilla",
    "csd_plain_table_static": "CSD static",
    "csd_plain_table": "CSD dynamic",
    "csd_plain_table_top5": "CSD top-keep 15K",
}
COLORS = {
    "vanilla": "#6B7280",
    "csd_plain_table_static": "#60A5FA",
    "csd_plain_table": "#2563EB",
    "csd_plain_table_top5": "#7C3AED",
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
    return task.split(":")[0].split("|")[0]


def method_from_run_tag(tag: str) -> str:
    for method in [
        "csd_plain_table_static",
        "csd_plain_table_top5",
        "csd_plain_table",
        "vanilla",
    ]:
        if tag.startswith(method + "_"):
            return method
    return tag.split("_")[0]


def load_rows(*jsonl_paths: Path) -> list[dict]:
    by_task_method: dict[tuple[str, str], dict] = {}
    for path in jsonl_paths:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                tag = row.get("other", {}).get("run_tag", "")
                item = {
                    "task": task_slug(row["task"]),
                    "method": method_from_run_tag(tag),
                    "accuracy": row.get("accuracy"),
                    "throughput": row.get("throughput"),
                    "spec_success_rate": row.get("spec_success_rate"),
                    "latency": row.get("latency"),
                    "run_tag": tag,
                }
                by_task_method[(item["task"], item["method"])] = item
    return [
        by_task_method[key]
        for key in sorted(
            by_task_method,
            key=lambda item: (TASKS.index(item[0]), METHOD_ORDER.index(item[1])),
        )
    ]


def plot_metric(
    rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    *,
    percent: bool = False,
    baseline_line: float | None = None,
) -> None:
    by_task_method = {(r["task"], r["method"]): r for r in rows}
    available_methods = [
        m for m in METHOD_ORDER if any((task, m) in by_task_method for task in TASKS)
    ]
    fig, ax = plt.subplots(figsize=(13.5, 6.8), facecolor="#F8FAFC")
    ax.set_facecolor("#FFFFFF")
    x = np.arange(len(TASKS))
    width = min(0.16, 0.78 / max(1, len(available_methods)))
    offsets = (
        np.arange(len(available_methods)) - (len(available_methods) - 1) / 2
    ) * width

    for off, method in zip(offsets, available_methods):
        vals = []
        for task in TASKS:
            r = by_task_method.get((task, method))
            val = r.get(metric) if r else None
            if val is not None and percent:
                val *= 100
            vals.append(np.nan if val is None else val)
        bars = ax.bar(
            x + off,
            vals,
            width=width * 0.92,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=1.0,
            alpha=0.94,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            fmt = (
                "{:.1f}"
                if percent
                else (
                    "{:.2f}"
                    if metric == "speedup_vs_vanilla"
                    else "{:.2f}" if metric.endswith("rate") else "{:.0f}"
                )
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#111827",
            )

    if baseline_line is not None:
        ax.axhline(
            baseline_line, color="#94A3B8", linewidth=1.2, linestyle="--", zorder=2
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
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=min(5, len(available_methods)),
        frameon=False,
    )
    ax.text(
        0.01,
        -0.19,
        "SGLang CSD LightEval comparison | Qwen-style light benchmark chart",
        transform=ax.transAxes,
        fontsize=9,
        color="#64748B",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--comparison-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.baseline_jsonl), Path(args.comparison_jsonl))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_task_method = {(r["task"], r["method"]): r for r in rows}
    vanilla_throughput = {
        task: by_task_method[(task, "vanilla")]["throughput"]
        for task in TASKS
        if (task, "vanilla") in by_task_method
    }
    for r in rows:
        base = vanilla_throughput.get(r["task"])
        r["speedup_vs_vanilla"] = (
            r["throughput"] / base if base and r.get("throughput") else None
        )

    plot_metric(
        rows,
        "accuracy",
        "Accuracy (%)",
        "Accuracy by Task and Decoding Method",
        out_dir / "accuracy",
        percent=True,
    )
    plot_metric(
        rows,
        "throughput",
        "Output throughput (tok/s)",
        "Throughput by Task and Decoding Method",
        out_dir / "throughput",
    )
    plot_metric(
        rows,
        "spec_success_rate",
        "Speculative success rate (%)",
        "Speculative Success Rate by Task and Method",
        out_dir / "spec_success_rate",
        percent=True,
    )
    plot_metric(
        rows,
        "speedup_vs_vanilla",
        "Speedup vs Vanilla (×)",
        "Throughput Speedup Relative to Vanilla",
        out_dir / "speedup_vs_vanilla",
        baseline_line=1.0,
    )

    (out_dir / "plot_data_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
