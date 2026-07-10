#!/usr/bin/env python3
"""Focused CPU/GIL profile for the production CSD rebuild function."""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
CSD_RUNTIME_PATH = REPO_ROOT / "python/sglang/srt/speculative/csd_runtime.py"


def load_csd_runtime_module():
    spec = importlib.util.spec_from_file_location(
        "csd_runtime_cpu_profile", CSD_RUNTIME_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {CSD_RUNTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-path", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--delta-size", type=int, default=4096)
    parser.add_argument("--freq-threshold", type=int, default=6)
    parser.add_argument("--max-probe", type=int, default=16)
    parser.add_argument("--heartbeat-sleep-ms", type=float, default=0.5)
    parser.add_argument("--native-extension", default=None)
    parser.add_argument("--stateful-native", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.native_extension:
        torch.ops.load_library(args.native_extension)
    csd = load_csd_runtime_module()
    store = csd.CSDTableStore.load(args.table_path)

    # Initialize the same frequency membership cache used by online rebuild.
    initial_keys = store.filtered_keys(args.freq_threshold)
    delta_source = list(store.entries)[: args.delta_size]
    counts = Counter(delta_source)
    native_builder = None
    native_pairs = None
    if args.stateful_native:
        native_builder = torch.classes.sgl_kernel.CSDTableBuilder(
            torch.tensor(list(store.entries), dtype=torch.int64),
            torch.tensor(
                [entry.freq for entry in store.entries.values()], dtype=torch.int64
            ),
            args.freq_threshold,
        )
        native_pairs = torch.tensor(delta_source, dtype=torch.int64)

    heartbeat_gaps_ms: list[float] = []
    rebuild_ms: list[float] = []
    heartbeat_count = 0
    previous_heartbeat = time.perf_counter()

    def rebuild_once():
        start = time.perf_counter()
        if native_builder is not None:
            table, num_entries, _ = native_builder.update_and_build(
                native_pairs,
                args.max_probe,
                0.5,
            )
            payload = csd.CSDHashTablePayload(
                keys=table,
                num_entries=num_entries,
                capacity=table.numel(),
                max_probe=args.max_probe,
            )
        else:
            payload = csd.build_csd_rebuild_payload_from_counts(
                store,
                counts,
                args.freq_threshold,
                max_probe=args.max_probe,
            )
        rebuild_ms.append((time.perf_counter() - start) * 1000)
        return payload

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="csd-rebuild") as pool:
        future = pool.submit(rebuild_once)
        completed = 0
        while completed < args.iterations:
            now = time.perf_counter()
            heartbeat_gaps_ms.append((now - previous_heartbeat) * 1000)
            previous_heartbeat = now
            heartbeat_count += 1
            time.sleep(args.heartbeat_sleep_ms / 1000)
            if future.done():
                payload = future.result()
                completed += 1
                if completed < args.iterations:
                    future = pool.submit(rebuild_once)

    elapsed = time.perf_counter() - start
    sorted_gaps = sorted(heartbeat_gaps_ms)
    p99_index = min(len(sorted_gaps) - 1, int(len(sorted_gaps) * 0.99))
    print(
        {
            "table_entries": len(store.entries),
            "initial_filtered_keys": len(initial_keys),
            "final_hash_entries": payload.num_entries,
            "iterations": args.iterations,
            "stateful_native": args.stateful_native,
            "elapsed_sec": round(elapsed, 3),
            "rebuild_mean_ms": round(statistics.mean(rebuild_ms), 3),
            "rebuild_max_ms": round(max(rebuild_ms), 3),
            "main_heartbeat_count": heartbeat_count,
            "main_heartbeat_p99_gap_ms": round(sorted_gaps[p99_index], 3),
            "main_heartbeat_max_gap_ms": round(max(sorted_gaps), 3),
        }
    )


if __name__ == "__main__":
    main()
