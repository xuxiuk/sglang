#!/usr/bin/env python3
"""Analyze calibration frequencies of judge-confirmed speculative misrejections."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LABELS = ("PROVEN_EQUIVALENT", "DRAFT_BETTER")
EVENT_METRICS = (
    "generated_position",
    "accepted_drafts_before_rejection",
    "calibration_frequency",
    "draft_target_probability",
    "residual_target_probability",
    "draft_residual_probability_ratio",
    "target_top1_probability",
    "draft_top1_probability_ratio",
    "target_top1_top2_margin",
    "draft_target_logit",
    "residual_target_logit",
    "max_target_logit",
    "entropy_raw",
    "entropy_norm_vocab",
    "effective_support_size",
    "table_hit",
    "probability_ratio_gate_passed",
    "entropy_gate_passed",
    "would_force_accept",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Join judged rejection events with a CSD calibration frequency store "
            "and measure whether confirmed misrejections are head-concentrated or long-tailed."
        )
    )
    parser.add_argument("--judge", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--positive-labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Mapped judge labels treated as explicit misrejection evidence.",
    )
    parser.add_argument("--freq-threshold", type=int, default=6)
    parser.add_argument("--top-pairs", type=int, default=100)
    return parser.parse_args()


def pair_key(lhs: int, rhs: int) -> int:
    return ((int(lhs) & 0xFFFFFFFF) << 32) | (int(rhs) & 0xFFFFFFFF)


def load_table(path: Path):
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        metadata = {}
    else:
        payload = json.loads(path.read_text())
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        records = payload.get("entries", []) if isinstance(payload, dict) else payload
    counts = collections.Counter()
    for row in records:
        if row.get("type") == "metadata":
            metadata.update(row.get("metadata", {}))
            continue
        key = row.get("key")
        if key is None:
            key = pair_key(
                row.get("lhs_token", row.get("draft_token")),
                row.get("rhs_token", row.get("residual_token")),
            )
        counts[int(key)] += int(row.get("freq", 1))
    return counts, metadata


def task_name(path: Path) -> str:
    # Expected form: <run>/<task>/judge/<file>.jsonl.
    return path.parent.parent.name if path.parent.name == "judge" else path.stem


def load_judgments(paths: list[Path], table_counts):
    rows = []
    mismatches = 0
    for path in paths:
        task = task_name(path)
        for line in path.read_text().splitlines():
            if not line:
                continue
            judged = json.loads(line)
            event = judged["source_event"]
            draft = int(event["draft_token_id"])
            residual = int(event["residual_token_id"])
            frequency = int(table_counts.get(pair_key(draft, residual), 0))
            traced_frequency = event.get("table_frequency")
            if traced_frequency is not None and int(traced_frequency) != frequency:
                mismatches += 1
            rows.append(
                {
                    "task": task,
                    "request_id": judged["request_id"],
                    "generated_position": int(judged["generated_position"]),
                    "label": judged["mapped_label"],
                    "draft_token_id": draft,
                    "residual_token_id": residual,
                    "pair_key": pair_key(draft, residual),
                    "calibration_frequency": frequency,
                    "accepted_drafts_before_rejection": event.get(
                        "accepted_drafts_before_rejection"
                    ),
                    "trace_table_frequency": traced_frequency,
                    "table_hit": event.get("table_hit"),
                    "probability_ratio_gate_passed": event.get(
                        "probability_ratio_gate_passed"
                    ),
                    "entropy_gate_passed": event.get("entropy_gate_passed"),
                    "would_force_accept": event.get("would_force_accept"),
                    "draft_target_probability": event.get("draft_target_probability"),
                    "residual_target_probability": event.get(
                        "residual_target_probability"
                    ),
                    "draft_residual_probability_ratio": event.get(
                        "draft_residual_probability_ratio"
                    ),
                    "target_top1_probability": event.get("target_top1_probability"),
                    "draft_top1_probability_ratio": event.get(
                        "draft_top1_probability_ratio"
                    ),
                    "target_top1_top2_margin": event.get(
                        "target_top1_top2_margin"
                    ),
                    "draft_target_logit": event.get("draft_target_logit"),
                    "residual_target_logit": event.get("residual_target_logit"),
                    "max_target_logit": event.get("max_target_logit"),
                    "entropy_raw": event.get("entropy_raw"),
                    "entropy_norm_vocab": event.get("entropy_norm_vocab"),
                    "effective_support_size": event.get("effective_support_size"),
                }
            )
    return rows, mismatches


def quantiles(values):
    if not values:
        return {str(q): None for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)}
    array = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.quantile(array, q)) for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)}


def rates(values, thresholds):
    if not values:
        return {str(t): None for t in thresholds}
    return {str(t): sum(v >= t for v in values) / len(values) for t in thresholds}


def concentration(selected_rows, fractions=(0.001, 0.01, 0.05, 0.1, 0.2)):
    occurrences = collections.Counter(row["pair_key"] for row in selected_rows)
    ranked = sorted(occurrences.values(), reverse=True)
    total = sum(ranked)
    unique = len(ranked)
    result = {}
    for fraction in fractions:
        keep = max(1, math.ceil(unique * fraction)) if unique else 0
        result[str(fraction)] = sum(ranked[:keep]) / total if total else None
    return occurrences, result


def percentile_ranks(values, reference):
    reference = np.sort(np.asarray(reference, dtype=np.int64))
    if not len(reference):
        return [0.0] * len(values)
    return [float(np.searchsorted(reference, value, side="right") / len(reference)) for value in values]


def finite_values(rows, metric):
    values = []
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def roc_auc(positive, negative):
    """Probability that a random positive value exceeds a random negative value."""
    if not len(positive) or not len(negative):
        return None
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    pos_rank_sum = ranks[: len(positive)].sum()
    return float(
        (pos_rank_sum - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def ks_distance(a, b):
    if not len(a) or not len(b):
        return None
    points = np.sort(np.unique(np.concatenate([a, b])))
    a_sorted, b_sorted = np.sort(a), np.sort(b)
    a_cdf = np.searchsorted(a_sorted, points, side="right") / len(a_sorted)
    b_cdf = np.searchsorted(b_sorted, points, side="right") / len(b_sorted)
    return float(np.max(np.abs(a_cdf - b_cdf)))


def metric_statistics(rows, selected, metrics):
    selected_ids = {
        (row["task"], row["request_id"], row["generated_position"]) for row in selected
    }
    other = [
        row
        for row in rows
        if (row["task"], row["request_id"], row["generated_position"])
        not in selected_ids
    ]
    result = {}
    for metric in metrics:
        positive = finite_values(selected, metric)
        negative = finite_values(other, metric)
        result[metric] = {
            "explicit_count": int(len(positive)),
            "other_count": int(len(negative)),
            "explicit_mean": float(positive.mean()) if len(positive) else None,
            "other_mean": float(negative.mean()) if len(negative) else None,
            "explicit_median": float(np.median(positive)) if len(positive) else None,
            "other_median": float(np.median(negative)) if len(negative) else None,
            "explicit_quantiles": quantiles(positive.tolist()),
            "other_quantiles": quantiles(negative.tolist()),
            "roc_auc_high_predicts_explicit": roc_auc(positive, negative),
            "ks_distance": ks_distance(positive, negative),
        }
    return result


def write_events(path, rows, table_percentiles, positive_labels):
    fields = list(rows[0]) + ["explicit_misrejection", "table_frequency_percentile"] if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, percentile in zip(rows, table_percentiles):
            writer.writerow(
                {
                    **row,
                    "explicit_misrejection": row["label"] in positive_labels,
                    "table_frequency_percentile": percentile,
                }
            )


def write_pairs(path, selected_rows, table_counts, top_pairs):
    occurrences = collections.Counter(row["pair_key"] for row in selected_rows)
    example = {row["pair_key"]: row for row in selected_rows}
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "rank",
            "draft_token_id",
            "residual_token_id",
            "calibration_frequency",
            "misrejection_events",
            "event_share",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        total = sum(occurrences.values())
        ranked = sorted(
            occurrences.items(),
            key=lambda item: (item[1], table_counts.get(item[0], 0)),
            reverse=True,
        )
        for rank, (key, count) in enumerate(ranked[:top_pairs], 1):
            row = example[key]
            writer.writerow(
                {
                    "rank": rank,
                    "draft_token_id": row["draft_token_id"],
                    "residual_token_id": row["residual_token_id"],
                    "calibration_frequency": table_counts.get(key, 0),
                    "misrejection_events": count,
                    "event_share": count / total if total else 0,
                }
            )


def plot_calibration_ranked_false_rejection_cdf(output, selected_rows, freq_threshold):
    """Rank false-rejection pairs by calibration frequency, then cumulate events."""
    event_counts = collections.Counter(row["pair_key"] for row in selected_rows)
    calibration_frequency = {
        row["pair_key"]: row["calibration_frequency"] for row in selected_rows
    }
    ranked_pairs = sorted(
        event_counts,
        key=lambda key: (calibration_frequency[key], event_counts[key], key),
        reverse=True,
    )
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    if ranked_pairs:
        event_total = sum(event_counts.values())
        x = np.arange(1, len(ranked_pairs) + 1) / len(ranked_pairs)
        y = np.cumsum([event_counts[key] for key in ranked_pairs]) / event_total
        axis.plot(x, y, color="#2F6BBD", linewidth=2.8)

        active_pair_count = sum(
            calibration_frequency[key] >= freq_threshold for key in ranked_pairs
        )
        if active_pair_count:
            active_x = active_pair_count / len(ranked_pairs)
            active_y = y[active_pair_count - 1]
            axis.axvline(active_x, color="#D1495B", linestyle="--", linewidth=1.8)
            axis.scatter([active_x], [active_y], color="#D1495B", zorder=3)
            axis.annotate(
                f"freq ≥ {freq_threshold}\n{active_pair_count:,} pairs cover {active_y:.1%} events",
                xy=(active_x, active_y),
                xytext=(min(active_x + 0.08, 0.72), max(active_y - 0.18, 0.08)),
                arrowprops={"arrowstyle": "->", "color": "#D1495B"},
                fontsize=10,
            )
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, math.ceil(len(ranked_pairs) * fraction))
            axis.scatter([count / len(ranked_pairs)], [y[count - 1]], s=22, color="#2F6BBD")
            axis.text(
                count / len(ranked_pairs),
                min(y[count - 1] + 0.035, 0.97),
                f"{fraction:.0%}: {y[count - 1]:.1%}",
                ha="center",
                fontsize=9,
            )
    axis.plot([0, 1], [0, 1], linestyle=":", color="0.65", linewidth=1.2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Cumulative fraction of false-rejection pairs\n(ranked by calibration frequency, high → low)")
    axis.set_ylabel("Cumulative fraction of false-rejection events")
    axis.set_title("Coverage of False Rejections by Calibration-Frequency Ranking")
    axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_all_metrics(output, rows, selected, metrics):
    selected_ids = {
        (row["task"], row["request_id"], row["generated_position"]) for row in selected
    }
    other = [
        row
        for row in rows
        if (row["task"], row["request_id"], row["generated_position"])
        not in selected_ids
    ]
    columns = 3
    rows_count = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(16, 3.65 * rows_count))
    axes = np.asarray(axes).reshape(-1)
    log_metrics = {
        "generated_position",
        "calibration_frequency",
        "effective_support_size",
    }
    for axis, metric in zip(axes, metrics):
        positive = finite_values(selected, metric)
        negative = finite_values(other, metric)
        combined = np.concatenate([positive, negative]) if len(positive) or len(negative) else np.array([])
        if not len(combined):
            axis.set_visible(False)
            continue
        if metric in log_metrics:
            transformed_positive = np.log10(positive + 1)
            transformed_negative = np.log10(negative + 1)
            xlabel = f"log10(1 + {metric})"
        else:
            transformed_positive = positive
            transformed_negative = negative
            xlabel = metric
        transformed = np.concatenate([transformed_positive, transformed_negative])
        lo, hi = np.quantile(transformed, [0.005, 0.995])
        if lo == hi:
            lo, hi = transformed.min() - 0.5, transformed.max() + 0.5
        bins = np.linspace(lo, hi, 36)
        axis.hist(
            transformed_negative,
            bins=bins,
            density=True,
            alpha=0.42,
            label="Other rejections",
            color="#7A8793",
        )
        axis.hist(
            transformed_positive,
            bins=bins,
            density=True,
            alpha=0.62,
            label="Explicit misrejections",
            color="#D1495B",
        )
        auc = roc_auc(positive, negative)
        ks = ks_distance(positive, negative)
        axis.set_title(f"{metric}\nAUC={auc:.3f}, KS={ks:.3f}")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Density")
    for axis in axes[len(metrics) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Trace metrics: explicit misrejections vs. other rejections", y=1.002, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_metric_statistics(path, statistics):
    fields = [
        "metric",
        "explicit_count",
        "other_count",
        "explicit_mean",
        "other_mean",
        "explicit_median",
        "other_median",
        "roc_auc_high_predicts_explicit",
        "ks_distance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, item in statistics.items():
            writer.writerow({"metric": metric, **{field: item[field] for field in fields[1:]}})


def main():
    args = parse_args()
    if args.freq_threshold < 1:
        raise ValueError("--freq-threshold must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_counts, metadata = load_table(args.calibration_table)
    rows, mismatches = load_judgments(args.judge, table_counts)
    positive = set(args.positive_labels)
    selected = [row for row in rows if row["label"] in positive]
    all_freq = [row["calibration_frequency"] for row in rows]
    selected_freq = [row["calibration_frequency"] for row in selected]
    table_freq = list(table_counts.values())
    percentiles = percentile_ranks(all_freq, table_freq)
    selected_percentiles = [p for row, p in zip(rows, percentiles) if row["label"] in positive]
    occurrences, concentration_rates = concentration(selected)
    statistics = metric_statistics(rows, selected, EVENT_METRICS)
    thresholds = [1, 3, 6, 10, 20, 50, 100, 500, 1000]

    by_task = {}
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        task_selected = [row for row in task_rows if row["label"] in positive]
        frequencies = [row["calibration_frequency"] for row in task_selected]
        by_task[task] = {
            "judged_events": len(task_rows),
            "explicit_misrejection_events": len(task_selected),
            "explicit_misrejection_rate": len(task_selected) / len(task_rows) if task_rows else None,
            "frequency_quantiles": quantiles(frequencies),
            "frequency_threshold_rates": rates(frequencies, thresholds),
        }

    summary = {
        "inputs": {
            "judge_files": [str(path.resolve()) for path in args.judge],
            "calibration_table": str(args.calibration_table.resolve()),
            "positive_labels": args.positive_labels,
            "active_frequency_threshold": args.freq_threshold,
        },
        "table": {
            "unique_pairs": len(table_counts),
            "total_frequency": sum(table_counts.values()),
            "active_pairs": sum(freq >= args.freq_threshold for freq in table_counts.values()),
            "frequency_quantiles": quantiles(table_freq),
            "metadata": metadata,
        },
        "overall": {
            "judged_events": len(rows),
            "explicit_misrejection_events": len(selected),
            "explicit_misrejection_rate": len(selected) / len(rows) if rows else None,
            "unique_explicit_misrejection_pairs": len(occurrences),
            "table_frequency_mismatch_with_trace": mismatches,
            "frequency_quantiles": quantiles(selected_freq),
            "frequency_percentile_in_full_table_quantiles": quantiles(selected_percentiles),
            "frequency_threshold_rates": rates(selected_freq, thresholds),
            "pair_concentration": concentration_rates,
        },
        "tasks": by_task,
        "metric_statistics": statistics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    write_events(args.output_dir / "events.csv", rows, percentiles, positive)
    write_pairs(args.output_dir / "top_pairs.csv", selected, table_counts, args.top_pairs)
    write_metric_statistics(args.output_dir / "metric_statistics.csv", statistics)
    plot_calibration_ranked_false_rejection_cdf(
        args.output_dir / "false_rejection_calibration_rank_cdf.png",
        selected,
        args.freq_threshold,
    )
    plot_all_metrics(
        args.output_dir / "all_trace_metric_distributions.png",
        rows,
        selected,
        EVENT_METRICS,
    )

    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
