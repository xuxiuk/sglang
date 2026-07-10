#!/usr/bin/env python3
"""Generate LongBench-v2 answers with SGLang and record throughput/spec metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from sglang.test.simple_eval_longbench_v2 import format_longbench_v2_question
from sglang.test.test_utils import add_common_sglang_args_and_parse
from transformers import AutoTokenizer


DEFAULT_DATA_JSON = (
    "/root/sglang/benchmark/csd/runs/longbench_cache/longbench_v2_data.json"
)


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    return value.strip("_") or "unknown"


def load_rows(path: Path) -> list[dict[str, Any]]:
    # The public LongBench-v2 JSON contains unescaped control characters in a
    # few long contexts, so strict=False is needed for local generation sweeps.
    return json.loads(path.read_text(encoding="utf-8"), strict=False)


def group_value(row: dict[str, Any], group_by: str) -> str:
    return str(row.get(group_by, "unknown") or "unknown")


def expand_groups(rows: list[dict[str, Any]], groups_arg: str, group_by: str) -> list[str]:
    available = []
    seen = set()
    for row in rows:
        value = group_value(row, group_by)
        if value not in seen:
            available.append(value)
            seen.add(value)

    if groups_arg == "all":
        return available

    by_slug = {slugify(value): value for value in available}
    groups = []
    for raw in [item.strip() for item in groups_arg.split(",") if item.strip()]:
        groups.append(by_slug.get(slugify(raw), raw))
    return groups


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
    response = requests.post(
        url,
        json={"text": prompt, "sampling_params": sampling},
        timeout=None,
    )
    response.raise_for_status()
    return response.json()


def meta_token_count(meta: dict[str, Any]) -> tuple[int, int | None]:
    completion_tokens = int(meta.get("completion_tokens", 0))
    verify_ct = meta.get("spec_verify_ct")
    if verify_ct is not None:
        verify_ct = int(verify_ct)
    return completion_tokens, verify_ct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json", default=DEFAULT_DATA_JSON)
    parser.add_argument("--groups", default="all")
    parser.add_argument("--group-by", choices=["domain", "sub_domain"], default="domain")
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--num-examples-per-group", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=240000)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--enable-thinking", type=str_to_bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = add_common_sglang_args_and_parse(parser)

    data_json = Path(args.data_json)
    rows = load_rows(data_json)
    groups = expand_groups(rows, args.groups, args.group_by)
    group_set = set(groups)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    selected = []
    prompts = []
    prompt_token_counts = []
    skipped_by_length: dict[str, int] = defaultdict(int)
    for group in groups:
        group_rows = [
            row for row in rows if group_value(row, args.group_by) == group
        ][args.offset :]
        kept = 0
        for row_idx, row in enumerate(group_rows, start=args.offset):
            prompt = format_longbench_v2_question(row)
            chat_prompt = build_prompt(
                tokenizer, prompt, args.system_prompt, args.enable_thinking
            )
            prompt_tokens = len(tokenizer.encode(chat_prompt))
            if args.max_prompt_tokens and prompt_tokens > args.max_prompt_tokens:
                skipped_by_length[group] += 1
                continue
            selected.append((group, row_idx, row))
            prompts.append(chat_prompt)
            prompt_token_counts.append(prompt_tokens)
            kept += 1
            if (
                args.num_examples_per_group is not None
                and kept >= args.num_examples_per_group
            ):
                break

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

    completion_tokens = []
    verify_cts = []
    per_group_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "num_requests": 0,
            "completion_tokens": [],
            "verify_cts": [],
            "prompt_tokens": [],
        }
    )
    answer_path = Path(args.answer_file)
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    with answer_path.open("w", encoding="utf-8") as fout:
        for (group, idx, row), ret, prompt_tokens in zip(
            selected, rets, prompt_token_counts
        ):
            completion, verify_ct = meta_token_count(ret["meta_info"])
            completion_tokens.append(completion)
            if verify_ct is not None:
                verify_cts.append(verify_ct)
            stats = per_group_stats[group]
            stats["num_requests"] += 1
            stats["completion_tokens"].append(completion)
            stats["prompt_tokens"].append(prompt_tokens)
            if verify_ct is not None:
                stats["verify_cts"].append(verify_ct)
            fout.write(
                json.dumps(
                    {
                        "group": group,
                        "index": idx,
                        "prediction": ret["text"],
                        "answer": row.get("answer"),
                        "domain": row.get("domain"),
                        "sub_domain": row.get("sub_domain"),
                        "difficulty": row.get("difficulty"),
                        "length": row.get("length"),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion,
                        "spec_verify_ct": verify_ct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_completion_tokens = sum(completion_tokens)
    throughput = total_completion_tokens / latency if latency > 0 else 0.0
    total_verify = sum(verify_cts)
    accept_length = (
        total_completion_tokens / total_verify if verify_cts and total_verify > 0 else 1.0
    )

    model_id = args.model_id or args.tokenizer_path
    result = {
        "task": "longbench_v2:" + ",".join(slugify(group) for group in groups),
        "model_id": model_id,
        "backend": args.backend,
        "num_gpus": None,
        "latency": round(latency, 3),
        "throughput": round(throughput, 3),
        "accept_length": round(accept_length, 3),
        "num_requests": len(selected),
        "total_completion_tokens": total_completion_tokens,
        "avg_completion_tokens": round(total_completion_tokens / len(selected), 3)
        if selected
        else 0.0,
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
            "groups": groups,
            "group_by": args.group_by,
            "num_examples_per_group": args.num_examples_per_group,
            "max_prompt_tokens": args.max_prompt_tokens,
            "skipped_by_length": dict(skipped_by_length),
            "parallel": args.parallel,
            "answer_file": str(answer_path),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "system_prompt": args.system_prompt,
            "enable_thinking": args.enable_thinking,
            "tokenizer_path": args.tokenizer_path,
            "data_json": str(data_json),
        },
    }

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        for group in groups:
            stats = per_group_stats[group]
            task_completion = stats["completion_tokens"]
            task_verify = stats["verify_cts"]
            task_prompt = stats["prompt_tokens"]
            task_total_completion = sum(task_completion)
            task_total_verify = sum(task_verify)
            task_accept_length = (
                task_total_completion / task_total_verify
                if task_verify and task_total_verify > 0
                else 1.0
            )
            task_result = {
                "task": "longbench_v2:" + slugify(group),
                "model_id": model_id,
                "backend": args.backend,
                "num_gpus": None,
                "latency": None,
                "throughput": None,
                "accept_length": round(task_accept_length, 3),
                "num_requests": stats["num_requests"],
                "total_completion_tokens": task_total_completion,
                "avg_completion_tokens": round(
                    task_total_completion / stats["num_requests"], 3
                )
                if stats["num_requests"]
                else 0.0,
                "max_completion_tokens": max(task_completion)
                if task_completion
                else 0,
                "max_new_token_hits": sum(
                    1 for count in task_completion if count >= args.max_new_tokens
                ),
                "avg_prompt_tokens": round(sum(task_prompt) / len(task_prompt), 3)
                if task_prompt
                else 0.0,
                "max_prompt_tokens": max(task_prompt) if task_prompt else 0,
                "other": {
                    "model_id": model_id,
                    "group": group,
                    "group_by": args.group_by,
                    "parent_task": result["task"],
                    "num_examples_per_group": args.num_examples_per_group,
                    "max_prompt_tokens": args.max_prompt_tokens,
                    "skipped_by_length": skipped_by_length.get(group, 0),
                    "parallel": args.parallel,
                    "answer_file": str(answer_path),
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "enable_thinking": args.enable_thinking,
                },
            }
            fout.write(json.dumps(task_result, ensure_ascii=False) + "\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
