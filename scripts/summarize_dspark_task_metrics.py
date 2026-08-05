#!/usr/bin/env python3
"""Diff per-DP /server_info snapshots for one isolated DSpark task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    final = json.loads(args.final.read_text())
    before = baseline.get("internal_states", [])
    after = final.get("internal_states", [])
    if not after or len(before) != len(after):
        raise RuntimeError(
            f"Mismatched DP snapshots: baseline={len(before)} final={len(after)}"
        )

    csd_keys = (
        "csd_lookup_hit_ct",
        "csd_forced_accept_ct",
        "csd_delta_pair_ct",
        "csd_rebuild_started_ct",
        "csd_rebuild_applied_ct",
        "csd_rebuild_wall_sec",
        "csd_dp_aggregate_started_ct",
        "csd_dp_aggregate_pair_ct",
    )
    rows = []
    total_accept = 0
    total_forward = 0
    csd_sums = {key: 0 for key in csd_keys}
    for rank, (lhs, rhs) in enumerate(zip(before, after, strict=True)):
        accept = int(rhs.get("spec_total_num_accept_tokens", 0)) - int(
            lhs.get("spec_total_num_accept_tokens", 0)
        )
        forward = int(rhs.get("spec_total_num_forward_ct", 0)) - int(
            lhs.get("spec_total_num_forward_ct", 0)
        )
        lhs_csd = lhs.get("csd_metrics") or {}
        rhs_csd = rhs.get("csd_metrics") or {}
        csd_delta = {
            key: rhs_csd.get(key, 0) - lhs_csd.get(key, 0) for key in csd_keys
        }
        for key, value in csd_delta.items():
            csd_sums[key] += value
        total_accept += accept
        total_forward += forward
        rows.append(
            {
                "dp_rank": rank,
                "spec_accept_tokens": accept,
                "spec_verify_count": forward,
                "avg_spec_accept_length": accept / forward if forward else None,
                "csd_delta": csd_delta,
            }
        )

    draft_tokens = int(after[0].get("speculative_num_draft_tokens") or 0)
    correct_drafts = total_accept - total_forward
    proposed_drafts = total_forward * max(draft_tokens - 1, 0)
    payload = {
        "dp_size": len(after),
        "speculative_num_draft_tokens": draft_tokens,
        "spec_accept_tokens": total_accept,
        "spec_verify_count": total_forward,
        "avg_spec_accept_length": (
            total_accept / total_forward if total_forward else None
        ),
        "spec_num_correct_drafts": correct_drafts,
        "spec_num_proposed_drafts": proposed_drafts,
        "spec_accept_rate": (
            correct_drafts / proposed_drafts if proposed_drafts else None
        ),
        "csd_counter_sums_across_dp": csd_sums,
        "csd_forced_accept_over_lookup_hit": (
            csd_sums["csd_forced_accept_ct"] / csd_sums["csd_lookup_hit_ct"]
            if csd_sums["csd_lookup_hit_ct"]
            else None
        ),
        "per_dp": rows,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
