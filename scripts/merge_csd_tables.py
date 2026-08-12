#!/usr/bin/env python3
"""Merge rank-sharded CSD frequency tables without importing SGLang."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {output}; pass --force explicitly.")

    counts: Counter[int] = Counter()
    metadata = {"content": "csd_pair_frequency", "merged_from": []}
    for name in args.inputs:
        path = Path(name)
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        metadata["merged_from"].append(str(path))
        for entry in payload.get("entries", []):
            counts[int(entry["key"])] += int(entry.get("freq", 1))

    entries = []
    for key, freq in sorted(counts.items()):
        entries.append(
            {
                "key": key,
                "lhs_token": key >> 32,
                "rhs_token": key & 0xFFFFFFFF,
                "freq": freq,
                "allow": True,
            }
        )
    metadata.update({"num_shards": len(args.inputs), "num_unique_pairs": len(entries)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "entries": entries}, file, ensure_ascii=False)
    print(f"merged {len(args.inputs)} shards, {len(entries)} pairs -> {output}")


if __name__ == "__main__":
    main()
