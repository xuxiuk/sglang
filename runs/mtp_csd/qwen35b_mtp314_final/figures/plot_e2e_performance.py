#!/usr/bin/env python3
"""Plot end-to-end throughput/speedup and aggregate draft acceptance rate."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

SOURCES = {
    "AIME 2025": (
        RUN_ROOT / "results/aime25_avg16/results/classic_tree_shape_sweep.jsonl",
        "aime25_avg|0",
    ),
    "Math500": (
        RUN_ROOT / "results/math500_avg4/results/classic_tree_shape_sweep.jsonl",
        "math_500|0",
    ),
    "LiveCodeBench v6": (
        RUN_ROOT / "results/lcb_avg4/results/classic_tree_shape_sweep.jsonl",
        "lcb:codegeneration_v6|0",
    ),
    "GSM8K": (
        RUN_ROOT / "results/gsm8k_avg4/results/classic_tree_shape_sweep.jsonl",
        "gsm8k_avg|0",
    ),
}

METHODS = {
    "Bare MTP": "eagle_",
    "+ Plain CSD": "plain_",
    "+ Dynamic CSD": "dynamic_",
    "+ Dynamic CSD\n+ Entropy Gate": "dynamic_entropy_p20_",
}

# Qwen-report-inspired palette: red baseline and a light-to-dark blue method family.
COLORS = ["#9CC9E2", "#6BAED6", "#3182BD", "#08519C"]


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def select_result(path: Path, task: str, prefix: str) -> dict:
    matches = []
    for row in load_rows(path):
        run_tag = row.get("other", {}).get("run_tag", "")
        is_method = run_tag.startswith(prefix)
        if prefix == "dynamic_" and run_tag.startswith("dynamic_entropy_"):
            is_method = False
        if row.get("task") == task and is_method:
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one result for task={task!r}, prefix={prefix!r} in "
            f"{path}, found {len(matches)}"
        )
    return matches[0]


def source_for(dataset: str, method: str) -> tuple[Path, str, str]:
    source, task = SOURCES[dataset]
    prefix = METHODS[method]
    if dataset == "Math500" and prefix == "dynamic_entropy_p20_":
        prefix = "dynamic_entropy_ignore_ratio_"
    return source, task, prefix


def main() -> None:
    datasets = list(SOURCES)
    methods = list(METHODS)
    throughput = np.zeros((len(methods), len(datasets)))
    acceptance = np.full_like(throughput, np.nan)
    accept_length = np.full_like(throughput, np.nan)
    csv_rows = []

    for dataset_index, dataset in enumerate(datasets):
        for method_index, method in enumerate(methods):
            source, task, prefix = source_for(dataset, method)
            row = select_result(source, task, prefix)
            other = row["other"]
            perf = other["performance"]
            spec = other.get("speculative_metrics", {})
            throughput[method_index, dataset_index] = float(
                perf["output_token_throughput"]
            )
            acceptance[method_index, dataset_index] = (
                float(spec["aggregate_spec_accept_rate"]) * 100.0
            )
            accept_length[method_index, dataset_index] = float(
                spec["aggregate_spec_accept_length"]
            )

            csv_rows.append(
                {
                    "dataset": dataset,
                    "method": method.replace("\n", " "),
                    "e2e_output_throughput_tok_s": (
                        f"{throughput[method_index, dataset_index]:.6f}"
                    ),
                    "speedup_vs_bare_mtp": "",  # filled after baseline is known
                    "aggregate_draft_acceptance_percent": (
                        f"{acceptance[method_index, dataset_index]:.6f}"
                    ),
                    "aggregate_accept_length": (
                        f"{accept_length[method_index, dataset_index]:.6f}"
                    ),
                    "source": str(source),
                }
            )

    bare_index = methods.index("Bare MTP")
    speedup = throughput / throughput[bare_index]
    normalized_method_indices = {
        method.replace("\n", " "): index for index, method in enumerate(methods)
    }
    for row in csv_rows:
        method_index = normalized_method_indices[row["method"]]
        dataset_index = datasets.index(row["dataset"])
        row["speedup_vs_bare_mtp"] = f"{speedup[method_index, dataset_index]:.6f}"

    csv_path = OUTPUT_DIR / "mtp_csd_e2e_performance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 23,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
        }
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 7.6))
    x = np.arange(len(datasets))
    width = 0.16

    # Absolute E2E throughput is the bar height; speedup is annotated on the bar.
    for method_index, (method, color) in enumerate(zip(methods, COLORS)):
        positions = x + (method_index - 1.5) * width
        bars = ax_left.bar(
            positions,
            throughput[method_index],
            width,
            color=color,
            edgecolor="white",
            linewidth=0.9,
        )
        for dataset_index, bar in enumerate(bars):
            value = throughput[method_index, dataset_index]
            factor = speedup[method_index, dataset_index]
            ax_left.text(
                bar.get_x() + bar.get_width() / 2,
                value + 65,
                f"{factor:.2f}×",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="medium",
                rotation=90,
            )

    ax_left.set_title("End-to-End Output Throughput", pad=18)
    ax_left.set_ylabel("Completion throughput (token/s)")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(datasets, rotation=0, ha="center")
    ax_left.set_ylim(0, throughput.max() * 1.20)
    ax_left.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.30)
    ax_left.set_axisbelow(True)

    spec_methods = methods
    spec_colors = COLORS
    spec_width = 0.19
    for local_index, (method, color) in enumerate(zip(spec_methods, spec_colors)):
        method_index = methods.index(method)
        positions = x + (local_index - 1.5) * spec_width
        bars = ax_right.bar(
            positions,
            acceptance[method_index],
            spec_width,
            color=color,
            edgecolor="white",
            linewidth=0.9,
        )
        for dataset_index, bar in enumerate(bars):
            rate = acceptance[method_index, dataset_index]
            ax_right.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.8,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="medium",
            )

    ax_right.set_title("Aggregate Draft Acceptance", pad=18)
    ax_right.set_ylabel("Accepted speculative positions (%)")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(datasets, rotation=0, ha="center")
    ax_right.set_ylim(0, max(100, np.nanmax(acceptance) * 1.17))
    ax_right.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.30)
    ax_right.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    fig.legend(
        handles,
        methods,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Qwen3.5-35B-A3B MTP + CSD End-to-End Performance "
        "(Max Running Requests = 48)",
        y=1.09,
        fontsize=26,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=3.0)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"mtp_csd_e2e_performance.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
