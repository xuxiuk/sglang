#!/usr/bin/env python3
"""Generate chat benchmark answers with SGLang and record throughput metrics.

Supported datasets:
- alpaca_eval: loads tatsu-lab/alpaca_eval/alpaca_eval.json
- arena_hard_v0.1 / arena_hard_v2.0: loads lmarena-ai/arena-hard-auto question.jsonl
- ifeval: loads google/IFEval train split

This script only runs model generation. Official judge scoring can be run later
with AlpacaEval or Arena-Hard-Auto using the emitted answer files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sglang.test.test_utils import add_common_sglang_args_and_parse
from transformers import AutoTokenizer


DEFAULT_SYSTEM_PROMPT = ""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset_rows(dataset: str, data_file: str | None) -> list[dict[str, Any]]:
    if data_file:
        path = Path(data_file)
        if path.suffix == ".jsonl":
            return read_jsonl(path)
        with open(path, "r", encoding="utf-8") as fin:
            return json.load(fin)

    if dataset == "alpaca_eval":
        path = hf_hub_download(
            repo_id="tatsu-lab/alpaca_eval",
            filename="alpaca_eval.json",
            repo_type="dataset",
        )
        return json.load(open(path, "r", encoding="utf-8"))

    arena_paths = {
        "arena_hard_v0.1": "data/arena-hard-v0.1/question.jsonl",
        "arena_hard_v2.0": "data/arena-hard-v2.0/question.jsonl",
    }
    if dataset in arena_paths:
        path = hf_hub_download(
            repo_id="lmarena-ai/arena-hard-auto",
            filename=arena_paths[dataset],
            repo_type="dataset",
        )
        return read_jsonl(path)

    if dataset == "ifeval":
        return [dict(row) for row in load_dataset("google/IFEval", split="train")]

    raise ValueError(f"Unsupported dataset: {dataset}")


def prompt_from_row(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "alpaca_eval":
        return row["instruction"]
    if dataset.startswith("arena_hard"):
        return row["prompt"]
    if dataset == "ifeval":
        return row["prompt"]
    raise ValueError(dataset)


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def meta_token_count(meta: dict[str, Any]) -> tuple[int, int | None]:
    completion_tokens = int(meta.get("completion_tokens", 0))
    verify_ct = meta.get("spec_verify_ct")
    if verify_ct is not None:
        verify_ct = int(verify_ct)
    return completion_tokens, verify_ct


def build_prompt(tokenizer: Any, prompt: str, system_prompt: str, enable_thinking: bool) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def call_generate(url: str, prompt: str, sampling: dict[str, Any]) -> dict[str, Any]:
    tic = time.perf_counter()
    response = requests.post(
        url,
        json={"text": prompt, "sampling_params": sampling},
        timeout=None,
    )
    latency = time.perf_counter() - tic
    response.raise_for_status()
    ret = response.json()
    ret["_request_latency"] = latency
    ret["_request_end_time"] = time.perf_counter()
    return ret


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def peak_completed_window_throughput(
    completion_latency_pairs: list[tuple[float, int]], window_s: float
) -> float:
    if not completion_latency_pairs:
        return 0.0
    peak = 0.0
    left = 0
    token_sum = 0
    for right, (end_time, tokens) in enumerate(completion_latency_pairs):
        token_sum += tokens
        while left <= right and completion_latency_pairs[left][0] < end_time - window_s:
            token_sum -= completion_latency_pairs[left][1]
            left += 1
        # Use the full sliding-window duration. These are completion-event tokens,
        # not server-side decode tokens; dividing by the distance between two
        # nearly simultaneous completions creates artificial spikes.
        elapsed = min(window_s, max(end_time, 1e-9))
        peak = max(peak, token_sum / elapsed)
    return peak


def completed_window_throughputs(
    completion_latency_pairs: list[tuple[float, int]], window_s: float
) -> list[float]:
    if not completion_latency_pairs:
        return []
    values = []
    left = 0
    token_sum = 0
    for right, (end_time, tokens) in enumerate(completion_latency_pairs):
        token_sum += tokens
        while left <= right and completion_latency_pairs[left][0] < end_time - window_s:
            token_sum -= completion_latency_pairs[left][1]
            left += 1
        elapsed = min(window_s, max(end_time, 1e-9))
        values.append(token_sum / elapsed)
    return values


def completed_token_buckets(
    completion_latency_pairs: list[tuple[float, int]], total_latency: float, window_s: float
) -> list[dict[str, Any]]:
    if total_latency <= 0:
        return []
    num_buckets = max(1, math.ceil(total_latency / window_s))
    buckets = [
        {
            "window_index": idx,
            "start_s": round(idx * window_s, 6),
            "end_s": round(min((idx + 1) * window_s, total_latency), 6),
            "completed_requests": 0,
            "completed_tokens": 0,
        }
        for idx in range(num_buckets)
    ]
    for end_time, tokens in completion_latency_pairs:
        idx = min(num_buckets - 1, max(0, int(end_time // window_s)))
        buckets[idx]["completed_requests"] += 1
        buckets[idx]["completed_tokens"] += tokens
    for bucket in buckets:
        duration = max(1e-9, bucket["end_s"] - bucket["start_s"])
        bucket["completed_throughput"] = round(bucket["completed_tokens"] / duration, 6)
    return buckets


def peak_prefix_throughput(
    completion_latency_pairs: list[tuple[float, int]], min_request_fraction: float = 0.1
) -> float:
    if not completion_latency_pairs:
        return 0.0
    min_count = max(1, int(len(completion_latency_pairs) * min_request_fraction))
    token_sum = 0
    peak = 0.0
    for idx, (end_time, tokens) in enumerate(completion_latency_pairs, start=1):
        token_sum += tokens
        if idx >= min_count and end_time > 0:
            peak = max(peak, token_sum / end_time)
    return peak


def write_outputs(
    dataset: str,
    output_path: Path,
    model_id: str,
    rows: list[dict[str, Any]],
    answers: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset == "alpaca_eval":
        payload = []
        for row, answer in zip(rows, answers):
            item = {
                "instruction": row["instruction"],
                "output": answer,
                "generator": model_id,
            }
            if "dataset" in row:
                item["dataset"] = row["dataset"]
            payload.append(item)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return

    if dataset.startswith("arena_hard"):
        with open(output_path, "w", encoding="utf-8") as fout:
            for row, answer in zip(rows, answers):
                record = {
                    "uid": row["uid"],
                    "ans_id": uuid.uuid4().hex,
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": {"answer": answer}},
                    ],
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    if dataset == "ifeval":
        with open(output_path, "w", encoding="utf-8") as fout:
            for row, answer in zip(rows, answers):
                record = dict(row)
                record["response"] = answer
                record["model_id"] = model_id
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    raise ValueError(dataset)


def write_request_metrics(
    output_path: Path,
    rows: list[dict[str, Any]],
    prompt_token_counts: list[int],
    completion_tokens: list[int],
    verify_cts: list[int | None],
    request_latencies: list[float],
    request_end_offsets: list[float],
    meta_infos: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows):
            prompt = row.get("instruction") or row.get("prompt") or ""
            completion = completion_tokens[idx]
            latency = request_latencies[idx]
            verify_ct = verify_cts[idx]
            record = {
                "index": idx,
                "prompt_preview": str(prompt)[:256],
                "prompt_tokens": prompt_token_counts[idx],
                "completion_tokens": completion,
                "request_latency": round(latency, 6),
                "request_end_offset": round(request_end_offsets[idx], 6),
                "request_throughput": round(completion / latency, 6)
                if latency > 0
                else 0.0,
                "spec_verify_ct": verify_ct,
                "accept_length": round(completion / verify_ct, 6)
                if verify_ct and verify_ct > 0
                else None,
            }
            for key in ("dataset", "id", "question_id", "uid"):
                if key in row:
                    record[key] = row[key]
            for key, value in meta_infos[idx].items():
                if key.startswith("csd_") and isinstance(value, (int, float, str, bool)):
                    record[key] = value
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_completed_windows(output_path: Path, windows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for window in windows:
            fout.write(json.dumps(window, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["alpaca_eval", "arena_hard_v0.1", "arena_hard_v2.0", "ifeval"],
        required=True,
    )
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--enable-thinking", type=str_to_bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    args = add_common_sglang_args_and_parse(parser)

    rows = load_dataset_rows(args.dataset, args.data_file)
    rows = rows[args.offset :]
    if args.num_examples is not None:
        rows = rows[: args.num_examples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    prompts = [
        build_prompt(
            tokenizer,
            prompt_from_row(args.dataset, row),
            args.system_prompt,
            args.enable_thinking,
        )
        for row in rows
    ]
    prompt_token_counts = [len(tokenizer.encode(prompt)) for prompt in prompts]

    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
    }
    if args.top_k is not None and args.top_k >= 0:
        sampling["top_k"] = args.top_k

    url = f"http://{args.host}:{args.port}/generate"
    tic = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        rets = list(executor.map(lambda item: call_generate(url, item, sampling), prompts))
    latency = time.perf_counter() - tic

    answers = [ret["text"] for ret in rets]
    meta_infos = [ret.get("meta_info", {}) for ret in rets]
    completion_tokens = []
    verify_cts = []
    per_request_verify_cts: list[int | None] = []
    request_latencies = []
    request_end_offsets = []
    for ret in rets:
        completion, verify_ct = meta_token_count(ret["meta_info"])
        completion_tokens.append(completion)
        per_request_verify_cts.append(verify_ct)
        request_latencies.append(float(ret.get("_request_latency") or 0.0))
        request_end_offsets.append(float(ret.get("_request_end_time") or tic) - tic)
        if verify_ct is not None:
            verify_cts.append(verify_ct)

    total_completion_tokens = sum(completion_tokens)
    throughput = total_completion_tokens / latency if latency > 0 else 0.0
    request_throughputs = [
        tokens / req_latency if req_latency > 0 else 0.0
        for tokens, req_latency in zip(completion_tokens, request_latencies)
    ]
    completion_latency_pairs = sorted(
        zip(request_end_offsets, completion_tokens), key=lambda item: item[0]
    )
    completion_latency_token_triples = sorted(
        zip(request_end_offsets, completion_tokens, request_latencies),
        key=lambda item: item[0],
    )
    head90_count = int(len(completion_latency_pairs) * 0.9)
    head90_count = max(1, head90_count) if completion_latency_pairs else 0
    head90_pairs = completion_latency_pairs[:head90_count]
    head90_time = head90_pairs[-1][0] if head90_pairs else 0.0
    head90_tokens = sum(tokens for _, tokens in head90_pairs)
    head90_throughput = head90_tokens / head90_time if head90_time > 0 else 0.0
    tail10_count = (
        max(1, len(completion_latency_token_triples) - head90_count)
        if completion_latency_token_triples
        else 0
    )
    tail10_triples = (
        completion_latency_token_triples[-tail10_count:] if tail10_count else []
    )
    tail10_tokens = sum(tokens for _, tokens, _ in tail10_triples)
    tail10_first_end = tail10_triples[0][0] if tail10_triples else latency
    tail10_time = max(0.0, latency - tail10_first_end)
    peak_prefix_tok_s = peak_prefix_throughput(completion_latency_pairs)
    peak_completed_30s_tok_s = peak_completed_window_throughput(
        completion_latency_pairs, 30.0
    )
    peak_completed_60s_tok_s = peak_completed_window_throughput(
        completion_latency_pairs, 60.0
    )
    window30_throughputs = completed_window_throughputs(completion_latency_pairs, 30.0)
    window60_throughputs = completed_window_throughputs(completion_latency_pairs, 60.0)
    completed_30s_windows = completed_token_buckets(
        completion_latency_pairs, latency, 30.0
    )
    completed_60s_windows = completed_token_buckets(
        completion_latency_pairs, latency, 60.0
    )
    completed_30s_tokens = [
        float(window["completed_tokens"]) for window in completed_30s_windows
    ]
    completed_60s_tokens = [
        float(window["completed_tokens"]) for window in completed_60s_windows
    ]
    accept_length = (
        total_completion_tokens / sum(verify_cts)
        if verify_cts and sum(verify_cts) > 0
        else 1.0
    )

    model_id = args.model_id or args.tokenizer_path
    answer_path = Path(args.answer_file)
    write_outputs(args.dataset, answer_path, model_id, rows, answers)
    request_metrics_path = answer_path.with_name(
        answer_path.name + ".request_metrics.jsonl"
    )
    write_request_metrics(
        request_metrics_path,
        rows,
        prompt_token_counts,
        completion_tokens,
        per_request_verify_cts,
        request_latencies,
        request_end_offsets,
        meta_infos,
    )
    completed_windows_path = answer_path.with_name(
        answer_path.name + ".completed_windows.jsonl"
    )
    write_completed_windows(
        completed_windows_path,
        [
            {"window_s": 30, **window}
            for window in completed_30s_windows
        ]
        + [
            {"window_s": 60, **window}
            for window in completed_60s_windows
        ],
    )

    result = {
        "task": args.dataset,
        "model_id": model_id,
        "backend": args.backend,
        "num_gpus": None,
        "latency": round(latency, 3),
        "throughput": round(throughput, 3),
        "head90_throughput": round(head90_throughput, 3),
        "head90_request_count": head90_count,
        "head90_completion_tokens": head90_tokens,
        "tail10_request_count": tail10_count,
        "tail10_completion_tokens": tail10_tokens,
        "tail10_token_share": round(
            tail10_tokens / total_completion_tokens, 6
        )
        if total_completion_tokens > 0
        else 0.0,
        "tail10_time": round(tail10_time, 3),
        "tail10_time_share": round(tail10_time / latency, 6)
        if latency > 0
        else 0.0,
        "tail10_avg_completion_tokens": round(tail10_tokens / tail10_count, 3)
        if tail10_count
        else 0.0,
        "tail10_avg_request_latency": round(
            sum(req_latency for _, _, req_latency in tail10_triples) / tail10_count,
            3,
        )
        if tail10_count
        else 0.0,
        "peak_prefix_throughput": round(peak_prefix_tok_s, 3),
        "peak_completed_30s_throughput": round(peak_completed_30s_tok_s, 3),
        "peak_completed_60s_throughput": round(peak_completed_60s_tok_s, 3),
        "p50_completed_30s_throughput": round(percentile(window30_throughputs, 0.50) or 0.0, 3),
        "p90_completed_30s_throughput": round(percentile(window30_throughputs, 0.90) or 0.0, 3),
        "p95_completed_30s_throughput": round(percentile(window30_throughputs, 0.95) or 0.0, 3),
        "p50_completed_60s_throughput": round(percentile(window60_throughputs, 0.50) or 0.0, 3),
        "p90_completed_60s_throughput": round(percentile(window60_throughputs, 0.90) or 0.0, 3),
        "p95_completed_60s_throughput": round(percentile(window60_throughputs, 0.95) or 0.0, 3),
        "completed_30s_window_count": len(completed_30s_windows),
        "max_completed_30s_tokens": int(max(completed_30s_tokens))
        if completed_30s_tokens
        else 0,
        "p50_completed_30s_tokens": round(percentile(completed_30s_tokens, 0.50) or 0.0, 3),
        "p90_completed_30s_tokens": round(percentile(completed_30s_tokens, 0.90) or 0.0, 3),
        "p95_completed_30s_tokens": round(percentile(completed_30s_tokens, 0.95) or 0.0, 3),
        "completed_60s_window_count": len(completed_60s_windows),
        "max_completed_60s_tokens": int(max(completed_60s_tokens))
        if completed_60s_tokens
        else 0,
        "p50_completed_60s_tokens": round(percentile(completed_60s_tokens, 0.50) or 0.0, 3),
        "p90_completed_60s_tokens": round(percentile(completed_60s_tokens, 0.90) or 0.0, 3),
        "p95_completed_60s_tokens": round(percentile(completed_60s_tokens, 0.95) or 0.0, 3),
        "avg_request_latency": round(sum(request_latencies) / len(request_latencies), 3)
        if request_latencies
        else 0.0,
        "p50_request_latency": round(percentile(request_latencies, 0.50) or 0.0, 3),
        "p90_request_latency": round(percentile(request_latencies, 0.90) or 0.0, 3),
        "p95_request_latency": round(percentile(request_latencies, 0.95) or 0.0, 3),
        "p99_request_latency": round(percentile(request_latencies, 0.99) or 0.0, 3),
        "avg_request_throughput": round(
            sum(request_throughputs) / len(request_throughputs), 3
        )
        if request_throughputs
        else 0.0,
        "p50_request_throughput": round(percentile(request_throughputs, 0.50) or 0.0, 3),
        "p90_request_throughput": round(percentile(request_throughputs, 0.90) or 0.0, 3),
        "p95_request_throughput": round(percentile(request_throughputs, 0.95) or 0.0, 3),
        "p99_request_throughput": round(percentile(request_throughputs, 0.99) or 0.0, 3),
        "accept_length": round(accept_length, 3),
        "num_requests": len(rows),
        "total_completion_tokens": total_completion_tokens,
        "avg_completion_tokens": round(
            total_completion_tokens / len(rows), 3
        )
        if rows
        else 0.0,
        "p50_completion_tokens": round(percentile([float(x) for x in completion_tokens], 0.50) or 0.0, 3),
        "p90_completion_tokens": round(percentile([float(x) for x in completion_tokens], 0.90) or 0.0, 3),
        "p95_completion_tokens": round(percentile([float(x) for x in completion_tokens], 0.95) or 0.0, 3),
        "p99_completion_tokens": round(percentile([float(x) for x in completion_tokens], 0.99) or 0.0, 3),
        "max_completion_tokens": max(completion_tokens) if completion_tokens else 0,
        "max_new_token_hits": sum(
            1 for count in completion_tokens if count >= args.max_new_tokens
        ),
        "avg_prompt_tokens": round(sum(prompt_token_counts) / len(prompt_token_counts), 3)
        if prompt_token_counts
        else 0.0,
        "max_prompt_tokens": max(prompt_token_counts) if prompt_token_counts else 0,
        "other": {
            "model_id": model_id,
            "data_file": args.data_file,
            "parallel": args.parallel,
            "answer_file": str(answer_path),
            "request_metrics_file": str(request_metrics_path),
            "completed_windows_file": str(completed_windows_path),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "enable_thinking": args.enable_thinking,
            "tokenizer_path": args.tokenizer_path,
        },
    }

    with open(args.result_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        "#examples: {n}, Throughput: {thr:.2f} token/s, Acceptance length: {acc:.2f}, "
        "avg completion: {avg:.1f}, max_new hits: {hits}".format(
            n=len(rows),
            thr=throughput,
            acc=accept_length,
            avg=result["avg_completion_tokens"],
            hits=result["max_new_token_hits"],
        )
    )


if __name__ == "__main__":
    main()
