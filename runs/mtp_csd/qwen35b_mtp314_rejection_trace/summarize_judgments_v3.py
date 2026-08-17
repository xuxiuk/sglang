#!/usr/bin/env python3
import argparse
import collections
import json
from pathlib import Path


def metrics(rows):
    total = len(rows)
    mapped = [row["mapped_judgment"] for row in rows]
    counts = collections.Counter(
        (item["draft_validity"], item["residual_validity"], item["path_relation"])
        for item in mapped
    )
    derived = {
        "validated_draft": sum(x["draft_validity"] == "PROVEN_VALID" for x in mapped),
        "strong_draft_only_evidence": sum(
            x["draft_validity"] == "PROVEN_VALID" and x["residual_validity"] == "PROVEN_INVALID"
            for x in mapped
        ),
        "both_valid": sum(
            x["draft_validity"] == "PROVEN_VALID" and x["residual_validity"] == "PROVEN_VALID"
            for x in mapped
        ),
        "distinct_valid_path": sum(
            x["draft_validity"] == "PROVEN_VALID"
            and x["residual_validity"] == "PROVEN_VALID"
            and x["path_relation"] == "DISTINCT_BUT_VALID"
            for x in mapped
        ),
        "equivalent_path_lower_bound": sum(
            x["draft_validity"] == "PROVEN_VALID"
            and x["residual_validity"] == "PROVEN_VALID"
            and x["same_correct_outcome"] == "YES"
            and x["path_relation"] == "EQUIVALENT"
            for x in mapped
        ),
        "plausible_unproven_draft": sum(x["draft_validity"] == "PLAUSIBLE_UNPROVEN" for x in mapped),
    }
    return {
        "events": total,
        "derived_counts": derived,
        "derived_rates": {key: value / total if total else None for key, value in derived.items()},
        "joint_counts": {"|".join(key): value for key, value in sorted(counts.items())},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    args = p.parse_args()
    tasks, all_rows = {}, []
    for path in args.inputs:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        tasks[path.parent.parent.name] = metrics(rows)
        all_rows.extend(rows)
    prompt_version = all_rows[0].get("prompt_version", "UNKNOWN") if all_rows else "UNKNOWN"
    summary = {"prompt_version": prompt_version, "tasks": tasks, "overall": metrics(all_rows)}
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = [f"# {prompt_version} rejection-judge summary", "", "| Task | Events | Validated draft | Both valid | Distinct valid | Equivalent lower bound | Plausible draft |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task, item in [*tasks.items(), ("Overall", summary["overall"])]:
        rate = item["derived_rates"]
        lines.append(f"| {task} | {item['events']} | {rate['validated_draft']:.2%} | {rate['both_valid']:.2%} | {rate['distinct_valid_path']:.2%} | {rate['equivalent_path_lower_bound']:.2%} | {rate['plausible_unproven_draft']:.2%} |")
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
