#!/usr/bin/env python3
"""Plot the MTP and DSpark CSD calibration frequency tails."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_MTP = Path(
    "/root/sglang/benchmark/csd/assets/calibration/"
    "csd_table_redpajama_logits_ungated_6domains_n1000_"
    "Qwen3.5-35B-A3B_mtp_EAGLE_steps5_topk1_draft5_"
    "temp1.0_ratio0.01_maxnew1024.json"
)
DEFAULT_DSPARK = Path(
    "/root/sglang-dspark-csd/runs/dspark_csd/formal_redpajama_20260724/"
    "tables/dspark_csd_merged.json"
)
DEFAULT_OUTPUT = Path(
    "/root/sglang-dspark-csd/docs/_static/image/"
    "calibration_mtp_vs_dspark_long_tail.png"
)


def load_frequencies(path: Path) -> list[int]:
    with path.open() as f:
        data = json.load(f)
    return [int(entry["freq"]) for entry in data["entries"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtp-table", type=Path, default=DEFAULT_MTP)
    parser.add_argument("--dspark-table", type=Path, default=DEFAULT_DSPARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    thresholds = [1, 2, 3, 6, 10, 20, 50, 100]
    series = {
        "MTP": load_frequencies(args.mtp_table),
        "DSpark": load_frequencies(args.dspark_table),
    }
    colors = {"MTP": "#2f6fd6", "DSpark": "#d64b3f"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for name, frequencies in series.items():
        unique_total = len(frequencies)
        observation_total = sum(frequencies)
        unique_retained = [
            100 * sum(freq >= threshold for freq in frequencies) / unique_total
            for threshold in thresholds
        ]
        observation_coverage = [
            100
            * sum(freq for freq in frequencies if freq >= threshold)
            / observation_total
            for threshold in thresholds
        ]
        axes[0].plot(
            thresholds,
            unique_retained,
            marker="o",
            linewidth=2.2,
            label=name,
            color=colors[name],
        )
        axes[1].plot(
            thresholds,
            observation_coverage,
            marker="o",
            linewidth=2.2,
            label=name,
            color=colors[name],
        )

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(thresholds)
        axis.set_xticklabels([str(value) for value in thresholds])
        axis.axvline(6, color="#777777", linestyle="--", linewidth=1)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Minimum pair frequency")
        axis.legend()

    axes[0].set_ylabel("Retained unique pairs (%)")
    axes[0].set_title("Table size after frequency filtering")
    axes[1].set_ylabel("Covered calibration observations (%)")
    axes[1].set_title("Observation coverage of retained pairs")
    axes[0].annotate("threshold = 6", (6, 3.26), xytext=(8, 13), arrowprops={"arrowstyle": "->"})
    axes[1].annotate("MTP 44.68%", (6, 44.68), xytext=(8, 57), arrowprops={"arrowstyle": "->"})
    axes[1].annotate("DSpark 30.68%", (6, 30.68), xytext=(8, 18), arrowprops={"arrowstyle": "->"})
    fig.suptitle("CSD calibration frequency tail: MTP vs DSpark", fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
