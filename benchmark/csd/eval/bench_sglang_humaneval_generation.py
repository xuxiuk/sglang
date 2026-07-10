#!/usr/bin/env python3
"""Generate HumanEval completions with SGLang and record throughput/spec metrics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from sglang.test.test_utils import add_common_sglang_args_and_parse
from transformers import AutoTokenizer


INSTRUCTION = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Return only the Python code for the function body "
    "and any helper code you need.\n\n"
)


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def build_prompt(
    tokenizer: Any, prompt: str, system_prompt: str, enable_thinking: bool
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": INSTRUCTION + prompt})
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
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-samples-per-task", type=int, default=10)
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

    rows = [dict(row) for row in load_dataset("openai/openai_humaneval", split="test")]
    rows = rows[args.offset :]
    if args.num_examples is not None:
        rows = rows[: args.num_examples]

    expanded = []
    for row in rows:
        for sample_idx in range(args.num_samples_per_task):
            expanded.append((row, sample_idx))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    prompts = [
        build_prompt(tokenizer, row["prompt"], args.system_prompt, args.enable_thinking)
        for row, _ in expanded
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

    completion_tokens = []
    verify_cts = []
    answer_path = Path(args.answer_file)
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    with answer_path.open("w", encoding="utf-8") as fout:
        for idx, ((row, sample_idx), ret, prompt_tokens) in enumerate(
            zip(expanded, rets, prompt_token_counts)
        ):
            completion, verify_ct = meta_token_count(ret["meta_info"])
            completion_tokens.append(completion)
            if verify_ct is not None:
                verify_cts.append(verify_ct)
            fout.write(
                json.dumps(
                    {
                        "index": idx,
                        "task_id": row["task_id"],
                        "sample_idx": sample_idx,
                        "prediction": ret["text"],
                        "prompt": row["prompt"],
                        "entry_point": row["entry_point"],
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion,
                        "spec_verify_ct": verify_ct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_completion_tokens = sum(completion_tokens)
    total_verify = sum(verify_cts)
    accept_length = (
        total_completion_tokens / total_verify if verify_cts and total_verify > 0 else 1.0
    )
    throughput = total_completion_tokens / latency if latency > 0 else 0.0
    model_id = args.model_id or args.tokenizer_path
    result = {
        "task": "humaneval_repeat",
        "model_id": model_id,
        "backend": args.backend,
        "num_gpus": None,
        "latency": round(latency, 3),
        "throughput": round(throughput, 3),
        "accept_length": round(accept_length, 3),
        "num_requests": len(expanded),
        "num_tasks": len(rows),
        "num_samples_per_task": args.num_samples_per_task,
        "total_completion_tokens": total_completion_tokens,
        "avg_completion_tokens": round(total_completion_tokens / len(expanded), 3)
        if expanded
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
        },
    }
    if args.result_file:
        result_path = Path(args.result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
