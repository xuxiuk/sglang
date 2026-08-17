#!/usr/bin/env python3
import argparse
import collections
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    args = p.parse_args()
    tasks = {}
    total = collections.Counter()
    for path in args.inputs:
        rows = [json.loads(x) for x in path.read_text().splitlines() if x]
        counts = collections.Counter(row["mapped_label"] for row in rows)
        decidable = len(rows) - counts["INSUFFICIENT_CONTEXT"]
        strict = counts["PROVEN_EQUIVALENT"] + counts["DRAFT_BETTER"]
        unproven = counts["BOTH_PLAUSIBLE_UNPROVEN"]
        tasks[path.parent.parent.name] = {
            "events": len(rows),
            "decidable_events": decidable,
            "labels": dict(sorted(counts.items())),
            "strict_evidence_count": strict,
            "strict_evidence_rate_all": strict / len(rows) if rows else None,
            "plausible_unproven_count": unproven,
            "plausible_unproven_rate_all": unproven / len(rows) if rows else None,
        }
        total.update(counts)
    total_events = sum(total.values())
    total_decidable = total_events - total["INSUFFICIENT_CONTEXT"]
    total_strict = total["PROVEN_EQUIVALENT"] + total["DRAFT_BETTER"]
    total_unproven = total["BOTH_PLAUSIBLE_UNPROVEN"]
    summary = {
        "metric_definitions": {
            "strict": "PROVEN_EQUIVALENT or DRAFT_BETTER divided by all sampled events",
            "unproven": "BOTH_PLAUSIBLE_UNPROVEN divided by all sampled events",
        },
        "tasks": tasks,
        "overall": {
            "events": total_events,
            "decidable_events": total_decidable,
            "labels": dict(sorted(total.items())),
            "strict_evidence_count": total_strict,
            "strict_evidence_rate_all": total_strict / total_events if total_events else None,
            "plausible_unproven_count": total_unproven,
            "plausible_unproven_rate_all": total_unproven / total_events if total_events else None,
        },
    }
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Strict CSD rejection-judge summary", "", "| Task | Events | Decidable | Strict evidence / all | Plausible unproven / all |", "|---|---:|---:|---:|---:|"]
    for task, item in tasks.items():
        strict = item["strict_evidence_rate_all"]
        unproven = item["plausible_unproven_rate_all"]
        lines.append(f"| {task} | {item['events']} | {item['decidable_events']} | {strict:.2%} | {unproven:.2%} |")
    if total_events:
        lines.append(f"| Overall | {total_events} | {total_decidable} | {total_strict / total_events:.2%} | {total_unproven / total_events:.2%} |")
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
