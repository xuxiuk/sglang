#!/usr/bin/env python3
"""Generate BFCL answers with SGLang and record throughput/spec metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import hf_hub_download
from sglang.test.test_utils import add_common_sglang_args_and_parse
from transformers import AutoTokenizer


BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

BFCL_FILES = {
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel": "BFCL_v3_parallel.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
    "java": "BFCL_v3_java.json",
    "javascript": "BFCL_v3_javascript.json",
    "sql": "BFCL_v3_sql.json",
    "rest": "BFCL_v3_rest.json",
    "live_simple": "BFCL_v3_live_simple.json",
    "live_multiple": "BFCL_v3_live_multiple.json",
    "live_parallel": "BFCL_v3_live_parallel.json",
    "live_parallel_multiple": "BFCL_v3_live_parallel_multiple.json",
    "live_relevance": "BFCL_v3_live_relevance.json",
    "live_irrelevance": "BFCL_v3_live_irrelevance.json",
}


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def expand_categories(categories_arg: str) -> list[str]:
    if categories_arg == "all":
        return list(BFCL_FILES)
    categories = [item.strip() for item in categories_arg.split(",") if item.strip()]
    unknown = [item for item in categories if item not in BFCL_FILES]
    if unknown:
        raise ValueError(f"Unsupported BFCL categories: {unknown}")
    return categories


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_category_rows(
    category: str, cache_dir: str | None, limit: int | None, offset: int
) -> list[dict[str, Any]]:
    filename = BFCL_FILES[category]
    local_dir = Path(
        cache_dir
        or os.environ.get(
            "BFCL_DATA_DIR",
            "/root/sglang/benchmark/csd/runs/bfcl_cache",
        )
    )
    local_path = local_dir / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        path = local_path
    else:
        try:
            path = Path(
                hf_hub_download(
                    repo_id=BFCL_REPO,
                    filename=filename,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
            )
        except Exception:
            if local_path.exists() and local_path.stat().st_size > 0:
                path = local_path
            else:
                raise
    rows = read_jsonl(path)[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def flatten_messages(question: Any) -> list[dict[str, str]]:
    # Most BFCL single-turn rows use [[{"role": "user", ...}]], while SQL uses
    # [{"role": "user", ...}]. Flatten one-element wrappers conservatively.
    value = question
    while (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    if not isinstance(value, list):
        raise ValueError(f"Unexpected BFCL question shape: {type(question).__name__}")
    messages = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Unexpected BFCL message item: {item!r}")
        messages.append(
            {
                "role": str(item.get("role", "user")),
                "content": str(item.get("content", "")),
            }
        )
    return messages


def normalize_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {}
    for key, item in value.items():
        if key == "type" and item == "dict":
            normalized[key] = "object"
        elif key == "type" and item == "any":
            normalized[key] = "string"
        else:
            normalized[key] = normalize_schema(item)
    return normalized


def normalize_functions(functions: Any) -> list[dict[str, Any]]:
    if not isinstance(functions, list):
        raise ValueError("BFCL function field is not a list")
    return [normalize_schema(function) for function in functions]


def build_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    functions: list[dict[str, Any]],
    system_prompt: str,
    enable_thinking: bool,
) -> str:
    prompt_messages = []
    if system_prompt:
        prompt_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.extend(messages)
    return tokenizer.apply_chat_template(
        prompt_messages,
        tools=functions,
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


def summarize_stats(
    task: str,
    model_id: str,
    backend: str,
    latency: float | None,
    completion_tokens: list[int],
    verify_cts: list[int],
    prompt_tokens: list[int],
    max_new_tokens: int,
    num_requests: int,
    other: dict[str, Any],
) -> dict[str, Any]:
    total_completion = sum(completion_tokens)
    total_verify = sum(verify_cts)
    accept_length = (
        total_completion / total_verify if verify_cts and total_verify > 0 else 1.0
    )
    throughput = (
        total_completion / latency if latency is not None and latency > 0 else None
    )
    return {
        "task": task,
        "model_id": model_id,
        "backend": backend,
        "num_gpus": None,
        "latency": round(latency, 3) if latency is not None else None,
        "throughput": round(throughput, 3) if throughput is not None else None,
        "accept_length": round(accept_length, 3),
        "num_requests": num_requests,
        "total_completion_tokens": total_completion,
        "avg_completion_tokens": round(total_completion / num_requests, 3)
        if num_requests
        else 0.0,
        "max_completion_tokens": max(completion_tokens) if completion_tokens else 0,
        "max_new_token_hits": sum(
            1 for count in completion_tokens if count >= max_new_tokens
        ),
        "avg_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 3)
        if prompt_tokens
        else 0.0,
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else 0,
        "other": other,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl-categories", required=True)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--num-examples-per-category", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
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
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = add_common_sglang_args_and_parse(parser)

    categories = expand_categories(args.bfcl_categories)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    examples = []
    prompts = []
    prompt_token_counts = []
    for category in categories:
        rows = load_category_rows(
            category, args.hf_cache_dir, args.num_examples_per_category, args.offset
        )
        for idx, row in enumerate(rows):
            messages = flatten_messages(row["question"])
            functions = normalize_functions(row["function"])
            prompt = build_prompt(
                tokenizer,
                messages,
                functions,
                args.system_prompt,
                args.enable_thinking,
            )
            examples.append((category, idx + args.offset, row))
            prompts.append(prompt)
            prompt_token_counts.append(len(tokenizer.encode(prompt)))

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
    per_category_stats: dict[str, dict[str, Any]] = defaultdict(
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
        for (category, idx, row), ret, prompt_tokens in zip(
            examples, rets, prompt_token_counts
        ):
            completion, verify_ct = meta_token_count(ret["meta_info"])
            completion_tokens.append(completion)
            if verify_ct is not None:
                verify_cts.append(verify_ct)
            stats = per_category_stats[category]
            stats["num_requests"] += 1
            stats["completion_tokens"].append(completion)
            stats["prompt_tokens"].append(prompt_tokens)
            if verify_ct is not None:
                stats["verify_cts"].append(verify_ct)
            fout.write(
                json.dumps(
                    {
                        "category": category,
                        "index": idx,
                        "id": row.get("id"),
                        "prediction": ret["text"],
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion,
                        "spec_verify_ct": verify_ct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    model_id = args.model_id or args.tokenizer_path
    common_other = {
        "model_id": model_id,
        "bfcl_categories": categories,
        "num_examples_per_category": args.num_examples_per_category,
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
        "bfcl_repo": BFCL_REPO,
    }
    result = summarize_stats(
        "bfcl:" + ",".join(categories),
        model_id,
        args.backend,
        latency,
        completion_tokens,
        verify_cts,
        prompt_token_counts,
        args.max_new_tokens,
        len(examples),
        common_other,
    )

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        for category in categories:
            stats = per_category_stats[category]
            task_result = summarize_stats(
                "bfcl:" + category,
                model_id,
                args.backend,
                None,
                stats["completion_tokens"],
                stats["verify_cts"],
                stats["prompt_tokens"],
                args.max_new_tokens,
                stats["num_requests"],
                {
                    **common_other,
                    "parent_task": result["task"],
                    "bfcl_category": category,
                },
            )
            fout.write(json.dumps(task_result, ensure_ascii=False) + "\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
