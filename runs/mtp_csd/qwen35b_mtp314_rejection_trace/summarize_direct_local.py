#!/usr/bin/env python3
"""Summarize V5 direct-local judgments overall, by task and frequency bucket."""

import argparse
import collections
import json
from pathlib import Path


def frequency_bucket(f):
    if f == 0:
        return "0"
    if f <= 2:
        return "1-2"
    if f <= 5:
        return "3-5"
    if f <= 9:
        return "6-9"
    if f <= 99:
        return "10-99"
    return ">=100"


def metrics(rows):
    relations = collections.Counter(x["mapped_judgment"]["relation"] for x in rows)
    safe = relations["LOCALLY_SUBSTITUTABLE"]
    bases = collections.Counter(
        x["judge"].get("equivalence_basis", "NOT_APPLICABLE") for x in rows
    )
    insufficient = relations["INSUFFICIENT_BOUNDARY"]
    decidable = len(rows) - insufficient
    return {
        "events": len(rows),
        "decidable_events": decidable,
        "relations": dict(sorted(relations.items())),
        "equivalence_bases": dict(sorted(bases.items())),
        "safe_events": safe,
        "safe_rate_all": safe / len(rows) if rows else None,
        "safe_rate_decidable": safe / decidable if decidable else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--error-input", type=Path)
    p.add_argument("--expected-events", type=int)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    errors = []
    if args.error_input and args.error_input.is_file():
        errors = [json.loads(line) for line in args.error_input.read_text().splitlines() if line]
    by_task = {}
    for task in sorted({x["task_name"] for x in rows}):
        by_task[task] = metrics([x for x in rows if x["task_name"] == task])
    by_frequency = {}
    for bucket in ["0", "1-2", "3-5", "6-9", "10-99", ">=100"]:
        subset = [
            x for x in rows
            if frequency_bucket(int(x["source_event"].get("table_frequency", 0))) == bucket
        ]
        by_frequency[bucket] = metrics(subset)
    hit = [x for x in rows if int(x["source_event"].get("table_frequency", 0)) >= 6]
    miss = [x for x in rows if int(x["source_event"].get("table_frequency", 0)) < 6]
    result = {
        "prompt_version": rows[0]["prompt_version"] if rows else None,
        "overall": metrics(rows),
        "processing": {
            "expected_events": args.expected_events,
            "successful_events": len(rows),
            "failed_attempt_records": len(errors),
            "unique_failed_sample_ids": len({x["sample_id"] for x in errors}),
        },
        "by_task": by_task,
        "by_calibration_frequency": by_frequency,
        "active_frequency_ge6": metrics(hit),
        "inactive_frequency_lt6": metrics(miss),
    }
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# V5 Direct Local Judge Summary",
        "",
        "| Group | Events | Decidable | Safe / all | Safe / decidable |",
        "|---|---:|---:|---:|---:|",
    ]
    groups = [("Overall", result["overall"])]
    groups += [(f"Task: {k}", v) for k, v in by_task.items()]
    groups += [(f"Calibration frequency: {k}", v) for k, v in by_frequency.items()]
    groups += [("Frequency < 6", result["inactive_frequency_lt6"])]
    groups += [("Frequency >= 6", result["active_frequency_ge6"])]
    for name, item in groups:
        all_rate = item["safe_rate_all"]
        dec_rate = item["safe_rate_decidable"]
        lines.append(
            f"| {name} | {item['events']} | {item['decidable_events']} | "
            f"{all_rate:.2%} | {dec_rate:.2%} |"
            if all_rate is not None and dec_rate is not None
            else f"| {name} | {item['events']} | {item['decidable_events']} | N/A | N/A |"
        )
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
