#!/usr/bin/env python3
"""Plot event-self and calibration-ranked tails for candidate false rejections."""

import argparse
import collections
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freq-threshold", type=int, default=6)
    return p.parse_args()


def curve(pair_events, order):
    total = sum(pair_events.values())
    x = np.arange(1, len(order) + 1) / len(order)
    y = np.cumsum([pair_events[key] for key in order]) / total
    return x, y


def main():
    cfg = args()
    pair_events = collections.Counter()
    calibration_frequency = {}
    candidate_events = 0
    judged_events = 0
    for path in cfg.judge:
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            judged_events += 1
            validity = row["mapped_judgment"]["draft_validity"]
            if validity not in {"PROVEN_VALID", "PLAUSIBLE_UNPROVEN"}:
                continue
            event = row["source_event"]
            key = (int(event["draft_token_id"]), int(event["residual_token_id"]))
            pair_events[key] += 1
            calibration_frequency[key] = int(event.get("table_frequency", 0))
            candidate_events += 1
    if not pair_events:
        raise RuntimeError("No candidate false-rejection events")

    self_order = sorted(
        pair_events,
        key=lambda key: (pair_events[key], calibration_frequency[key], key),
        reverse=True,
    )
    calibration_order = sorted(
        pair_events,
        key=lambda key: (calibration_frequency[key], pair_events[key], key),
        reverse=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    specs = [
        (axes[0], self_order, "Ranked by false-rejection recurrence", "False-rejection self concentration"),
        (axes[1], calibration_order, "Ranked by calibration frequency", "Calibration coverage of false rejections"),
    ]
    for axis, order, xlabel, title in specs:
        x, y = curve(pair_events, order)
        axis.plot(x, y, color="#2F6BBD", linewidth=2.8)
        axis.plot([0, 1], [0, 1], ":", color="0.65", linewidth=1.2)
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, math.ceil(len(order) * fraction))
            coverage = y[count - 1]
            axis.scatter([count / len(order)], [coverage], color="#2F6BBD", s=22)
            axis.text(
                count / len(order), min(coverage + 0.035, 0.97),
                f"{fraction:.0%}: {coverage:.1%}", ha="center", fontsize=9,
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Cumulative fraction of candidate false-rejection pairs\n" + xlabel)
        axis.set_title(title)
        axis.grid(alpha=0.18)

    active = [key for key in calibration_order if calibration_frequency[key] >= cfg.freq_threshold]
    if active:
        x, y = curve(pair_events, calibration_order)
        index = len(active) - 1
        axes[1].axvline(x[index], color="#D1495B", linestyle="--", linewidth=1.8)
        axes[1].scatter([x[index]], [y[index]], color="#D1495B", zorder=3)
        axes[1].annotate(
            f"freq ≥ {cfg.freq_threshold}\n{len(active):,} pairs cover {y[index]:.1%} events",
            xy=(x[index], y[index]), xytext=(min(x[index] + 0.08, 0.68), max(y[index] - 0.16, 0.08)),
            arrowprops={"arrowstyle": "->", "color": "#D1495B"}, fontsize=10,
        )
    axes[0].set_ylabel("Cumulative fraction of candidate false-rejection events")
    fig.suptitle(
        f"Candidate false rejections = PROVEN_VALID + PLAUSIBLE_UNPROVEN\n"
        f"{candidate_events:,} / {judged_events:,} events; {len(pair_events):,} unique pairs",
        fontsize=14,
    )
    fig.tight_layout()
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output, dpi=220, bbox_inches="tight")
    print(cfg.output)


if __name__ == "__main__":
    main()
