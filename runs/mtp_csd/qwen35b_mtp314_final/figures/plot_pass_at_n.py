#!/usr/bin/env python3
"""Plot the selected MTP/CSD Pass@n results used in the final report."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

SOURCES = {
    "AIME 2025\nPass@16": (
        RUN_ROOT / "results/aime25_avg16/results/classic_tree_shape_sweep.jsonl",
        "aime25_avg|0",
        "pass@k:k=16&n=16",
    ),
    "Math500\nPass@4": (
        RUN_ROOT / "results/math500_avg4/results/classic_tree_shape_sweep.jsonl",
        "math_500|0",
        "pass@k:k=4&n=4",
    ),
    "LiveCodeBench v6\nPass@4": (
        RUN_ROOT / "results/lcb_avg4/results/classic_tree_shape_sweep.jsonl",
        "lcb:codegeneration_v6|0",
        "codegen_pass@4",
    ),
    "GSM8K\nPass@4": (
        RUN_ROOT / "results/gsm8k_avg4/results/classic_tree_shape_sweep.jsonl",
        "gsm8k_avg|0",
        "pass@k:k=4&n=4",
    ),
}

METHODS = {
    "Auto": "auto_",
    "Bare MTP": "eagle_",
    "+ Plain CSD": "plain_",
    "+ Dynamic CSD": "dynamic_",
    "+ Dynamic CSD + Entropy Gate": "dynamic_entropy_p20_",
}

# Qwen3's report uses red for the non-thinking baseline and blue for the main
# model series. Here Auto is the non-speculative baseline; all speculative
# variants stay in one increasingly dark blue family.
COLORS = ["#E53935", "#9CC9E2", "#6BAED6", "#3182BD", "#08519C"]


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


def main() -> None:
    datasets = list(SOURCES)
    methods = list(METHODS)
    values = np.zeros((len(methods), len(datasets)))
    errors = np.zeros_like(values)

    csv_rows = []
    for dataset_index, dataset in enumerate(datasets):
        source, task, metric = SOURCES[dataset]
        for method_index, method in enumerate(methods):
            method_source = source
            method_prefix = METHODS[method]
            if dataset.startswith("Math500") and method_prefix == "dynamic_entropy_p20_":
                method_prefix = "dynamic_entropy_ignore_ratio_"
            row = select_result(method_source, task, method_prefix)
            metrics = row["other"]["metrics"]
            value = float(metrics[metric]) * 100.0
            stderr = float(metrics[f"{metric}_stderr"]) * 100.0
            values[method_index, dataset_index] = value
            errors[method_index, dataset_index] = stderr
            csv_rows.append(
                {
                    "dataset": dataset.replace("\n", " "),
                    "method": method.replace("\n", " "),
                    "metric": metric,
                    "pass_at_n_percent": f"{value:.6f}",
                    "stderr_percent": f"{stderr:.6f}",
                    "source": str(method_source),
                }
            )

    with (OUTPUT_DIR / "mtp_csd_pass_at_n.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
        }
    )

    def render(show_errors: bool, stem: str) -> None:
        fig, ax = plt.subplots(figsize=(14.5, 7.6))
        x = np.arange(len(datasets))
        width = 0.16
        for method_index, (method, color) in enumerate(zip(methods, COLORS)):
            positions = x + (method_index - 2.0) * width
            error_kwargs = (
                {
                    "yerr": errors[method_index],
                    "capsize": 4,
                    "error_kw": {"elinewidth": 1.4, "alpha": 0.72},
                }
                if show_errors
                else {}
            )
            bars = ax.bar(
                positions,
                values[method_index],
                width,
                label=method,
                color=color,
                edgecolor="white",
                linewidth=0.9,
                **error_kwargs,
            )
            for bar, value, stderr in zip(
                bars, values[method_index], errors[method_index]
            ):
                label_y = value + (stderr if show_errors else 0.0) + 0.28
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="medium",
                )

        ax.set_title("Qwen3.5-35B-A3B: Pass@n Accuracy Preservation", pad=58)
        ax.set_ylabel("Pass@n (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(78, 104 if show_errors else 101.5)
        ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.30)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.115),
            ncol=5,
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220)
        fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    render(False, "mtp_csd_pass_at_n")
    render(True, "mtp_csd_pass_at_n_with_stderr")


if __name__ == "__main__":
    main()
