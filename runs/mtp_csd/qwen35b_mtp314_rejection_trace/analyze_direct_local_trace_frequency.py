#!/usr/bin/env python3
"""Audit V5 direct-local labels against pooled and task-local trace frequency."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


BUCKETS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-5", 3, 5),
    ("6-9", 6, 9),
    ("10-99", 10, 99),
    (">=100", 100, None),
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def bucket(value: int) -> str:
    for name, low, high in BUCKETS:
        if value >= low and (high is None or value <= high):
            return name
    raise ValueError(value)


def metrics(rows):
    relations = collections.Counter(x["mapped_judgment"]["relation"] for x in rows)
    safe = relations["LOCALLY_SUBSTITUTABLE"]
    return {
        "events": len(rows),
        "safe_events": safe,
        "safe_rate": safe / len(rows) if rows else None,
        "relations": dict(sorted(relations.items())),
    }


def sample_view(row):
    source = row["source_event"]
    judge = row["judge"]
    return {
        "sample_id": row["sample_id"],
        "task": row["task_name"],
        "draft": row["draft_token_text"],
        "residual": row["residual_token_text"],
        "relation": row["mapped_judgment"]["relation"],
        "equivalence_basis": row["mapped_judgment"].get("equivalence_basis"),
        "task_effect_category": row["mapped_judgment"].get("task_effect_category"),
        "concrete_task_effect": row["mapped_judgment"].get("concrete_task_effect"),
        "calibration_frequency": int(source.get("table_frequency", 0)),
        "pooled_trace_frequency": row["pooled_trace_frequency"],
        "task_trace_frequency": row["task_trace_frequency"],
        "context_tail": row["context_text"][-600:],
        "draft_interpretation": row["mapped_judgment"].get("draft_interpretation"),
        "residual_interpretation": row["mapped_judgment"].get("residual_interpretation"),
        "evidence": judge.get("evidence"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-group", type=int, default=12)
    args = parser.parse_args()

    pooled = collections.Counter()
    task_local = collections.Counter()
    raw_events = 0
    for task_dir in sorted(x for x in args.trace_root.iterdir() if x.is_dir()):
        task = task_dir.name
        for path in sorted((task_dir / "trace").glob("events.rank*.jsonl")):
            for event in read_jsonl(path):
                pair = (int(event["draft_token_id"]), int(event["residual_token_id"]))
                pooled[pair] += 1
                task_local[(task, *pair)] += 1
                raw_events += 1

    contexts = {x["sample_id"]: x["context_text"] for x in read_jsonl(args.samples)}
    rows = list(read_jsonl(args.judged))
    for row in rows:
        pair = (int(row["draft_token_id"]), int(row["residual_token_id"]))
        row["pooled_trace_frequency"] = pooled[pair]
        row["task_trace_frequency"] = task_local[(row["task_name"], *pair)]
        row["context_text"] = contexts[row["sample_id"]]

    by_pooled = {}
    by_task_local = {}
    for name, _, _ in BUCKETS:
        by_pooled[name] = metrics(
            [x for x in rows if bucket(x["pooled_trace_frequency"]) == name]
        )
        by_task_local[name] = metrics(
            [x for x in rows if bucket(x["task_trace_frequency"]) == name]
        )

    safe = [x for x in rows if x["mapped_judgment"]["relation"] == "LOCALLY_SUBSTITUTABLE"]
    different = [x for x in rows if x["mapped_judgment"]["relation"] == "BOTH_VALID_DIFFERENT"]
    safe_pooled_singleton = [x for x in safe if x["pooled_trace_frequency"] == 1]
    safe_pooled_high = [x for x in safe if x["pooled_trace_frequency"] >= 100]
    safe_task_singleton = [x for x in safe if x["task_trace_frequency"] == 1]

    rng = random.Random(args.seed)

    def select(items):
        items = list(items)
        rng.shuffle(items)
        return [sample_view(x) for x in items[: args.sample_per_group]]

    result = {
        "definitions": {
            "calibration_frequency": "source_event.table_frequency from the external RedPajama calibration table",
            "pooled_trace_frequency": "directed (draft_token_id, residual_token_id) count across all three complete evaluation traces",
            "task_trace_frequency": "directed pair count within the event's own evaluation task",
        },
        "counts": {
            "raw_trace_events": raw_events,
            "pooled_unique_pairs": len(pooled),
            "task_pair_keys": len(task_local),
            "judged_events": len(rows),
            "safe_events": len(safe),
            "different_valid_events": len(different),
            "safe_pooled_singleton_events": len(safe_pooled_singleton),
            "safe_pooled_high_ge100_events": len(safe_pooled_high),
            "safe_task_singleton_events": len(safe_task_singleton),
        },
        "by_pooled_trace_frequency": by_pooled,
        "by_task_trace_frequency": by_task_local,
        "audit_samples": {
            "both_valid_different": select(different),
            "safe_pooled_frequency_1": select(safe_pooled_singleton),
            "safe_pooled_frequency_ge100": select(safe_pooled_high),
            "safe_task_frequency_1": select(safe_task_singleton),
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.output_md:
        titles = {
            "both_valid_different": "BOTH_VALID_DIFFERENT 随机样本",
            "safe_pooled_frequency_1": "LOCALLY_SUBSTITUTABLE：pooled frequency = 1",
            "safe_pooled_frequency_ge100": "LOCALLY_SUBSTITUTABLE：pooled frequency >= 100",
            "safe_task_frequency_1": "LOCALLY_SUBSTITUTABLE：task-local frequency = 1",
        }
        lines = [
            "# Direct Local Judge 随机样本册",
            "",
            f"抽样种子：`{args.seed}`。每组最多 `{args.sample_per_group}` 条。",
            "",
            "样本仅展示原始上下文和 Judge 输出，不加入人工正确/错误标签。",
            "`pooled` 是三任务合并 trace 频率，`task-local` 是当前任务频率，",
            "`calibration` 是外部 RedPajama calibration table frequency。",
            "",
        ]
        for group, samples in result["audit_samples"].items():
            lines += [f"## {titles[group]}", ""]
            for index, sample in enumerate(samples, 1):
                lines += [
                    f"### {index}. `{sample['draft']}` → `{sample['residual']}`",
                    "",
                    f"- Task: `{sample['task']}`",
                    f"- Calibration frequency: `{sample['calibration_frequency']}`",
                    f"- Pooled trace frequency: `{sample['pooled_trace_frequency']}`",
                    f"- Task-local trace frequency: `{sample['task_trace_frequency']}`",
                    f"- Relation: `{sample['relation']}`",
                    f"- Equivalence basis: `{sample['equivalence_basis']}`",
                    f"- Task effect: `{sample['task_effect_category']}` / "
                    f"`{sample['concrete_task_effect']}`",
                    "",
                    "Context tail（候选 token 紧接在末尾之后）：",
                    "",
                    "```text",
                    sample["context_tail"],
                    "```",
                    "",
                    f"- Draft interpretation: {sample['draft_interpretation']}",
                    f"- Residual interpretation: {sample['residual_interpretation']}",
                    f"- Judge evidence: {sample['evidence']}",
                    "",
                ]
        args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "audit_samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
