#!/usr/bin/env python3
"""Profile short CSD dynamic/entropy overhead with in-process SGLang Engine.

This intentionally does not run LightEval. It loads one method at a time, warms up
short requests, profiles a small request batch, and writes torch profiler traces
plus a small JSON summary.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SGLANG_SRC = Path(os.environ.get("SGLANG_SRC", REPO_ROOT / "python"))
if str(SGLANG_SRC) not in sys.path:
    sys.path.insert(0, str(SGLANG_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from sglang import Engine


ENTROPY_P20_THRESHOLD = 1.5638477802276611

DEFAULT_PROMPTS = [
    """请一步一步推理并给出最终答案。

问题：一个数列满足 a_1 = 2，a_{n+1} = 3a_n + 1。请计算 a_8，并解释你的推导过程。
""",
    """请完整推理。一个班有 40 名学生，其中 24 人参加数学社，18 人参加编程社，10 人两个社团都参加。请问至少参加一个社团的学生有多少人？
""",
    """Solve the problem step by step. If x + y = 17 and xy = 60, compute x^2 + y^2 and explain the derivation.
""",
    """请写出推导过程：从 1 到 100 的整数中，能被 3 或 5 整除的数的和是多少？
""",
]



def method_overrides(
    method: str,
    table_path: str,
    freq_threshold: int,
    prob_ratio: float,
    rebuild_threshold: int,
):
    base = {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 5,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 5,
    }
    if method == "eagle":
        return base

    if method not in {
        "plain",
        "plain_entropy_p20",
        "dynamic",
        "dynamic_no_ratio",
        "dynamic_no_ratio_huge",
        "p20_no_ratio",
        "p20_no_ratio_huge",
    }:
        raise ValueError(f"unknown method: {method}")

    base.update(
        {
            "speculative_csd_enabled": True,
            "speculative_csd_table_path": table_path,
            "speculative_csd_freq_threshold": freq_threshold,
            "speculative_csd_key_selection_strategy": "frequency",
            "speculative_csd_score_threshold": 0.0,
        "speculative_csd_prob_ratio": prob_ratio,
        "speculative_csd_rebuild_threshold": rebuild_threshold,
    }
    )
    if method in {
        "dynamic",
        "dynamic_no_ratio",
        "dynamic_no_ratio_huge",
        "p20_no_ratio",
        "p20_no_ratio_huge",
    }:
        base["speculative_csd_dynamic_update"] = True
    if method in {
        "dynamic_no_ratio",
        "dynamic_no_ratio_huge",
        "p20_no_ratio",
        "p20_no_ratio_huge",
    }:
        base["speculative_csd_dynamic_update_ignore_prob_ratio"] = True
    if method in {"dynamic_no_ratio_huge", "p20_no_ratio_huge"}:
        base["speculative_csd_rebuild_threshold"] = 1_000_000_000
    if method in {"plain_entropy_p20", "p20_no_ratio", "p20_no_ratio_huge"}:
        base["speculative_csd_force_accept_entropy_threshold"] = ENTROPY_P20_THRESHOLD
    return base


def make_engine(args, method: str) -> Engine:
    engine_args = {
        "model_path": args.model,
        "tokenizer_path": args.tokenizer or args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "device": "cuda",
        "random_seed": 1234,
        "load_format": "auto",
        "context_length": args.max_length,
        "dp_size": 1,
        "tp_size": args.tp_size,
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
        "schedule_policy": "fcfs",
        "chunked_prefill_size": 4096,
        "disable_radix_cache": True,
        "watchdog_timeout": args.watchdog_timeout,
        "mamba_scheduler_strategy": args.mamba_scheduler_strategy,
        "log_level": "warning",
        "port": args.port,
    }
    engine_args.update(
        method_overrides(
            method,
            table_path=args.csd_table_path,
            freq_threshold=args.csd_freq_threshold,
            prob_ratio=args.csd_prob_ratio,
            rebuild_threshold=args.csd_rebuild_threshold,
        )
    )
    return Engine(**engine_args)


def output_token_count_one(output) -> int | None:
    meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
    if isinstance(meta.get("completion_tokens"), int):
        return meta["completion_tokens"]
    if isinstance(meta.get("output_token_logprobs"), list):
        return len(meta["output_token_logprobs"])
    if isinstance(output, dict) and isinstance(output.get("output_ids"), list):
        return len(output["output_ids"])
    return None


def output_token_count(output) -> int | None:
    if isinstance(output, list):
        counts = [output_token_count_one(item) for item in output]
        if any(count is None for count in counts):
            return None
        return sum(counts)
    return output_token_count_one(output)


def build_prompts(args) -> list[str]:
    if args.prompt_file:
        payload = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            all_prompts = [
                item if isinstance(item, str) else item.get("instruction", "")
                for item in payload
            ]
            if args.prompt_indices:
                indices = [int(part) for part in args.prompt_indices.split(",") if part]
                prompts = [all_prompts[index] for index in indices]
            else:
                prompts = all_prompts
        elif isinstance(payload, dict):
            prompts = [payload.get("instruction", "")]
        else:
            raise ValueError(f"Unsupported prompt file payload: {type(payload).__name__}")
        prompts = [prompt for prompt in prompts if prompt]
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompt_file}")
    elif args.prompt:
        prompts = [args.prompt]
    else:
        prompts = DEFAULT_PROMPTS
    if args.num_profile_requests <= len(prompts):
        return prompts[: args.num_profile_requests]
    return [prompts[i % len(prompts)] for i in range(args.num_profile_requests)]


def run_method(args, method: str, run_dir: Path):
    method_dir = run_dir / method
    trace_dir = method_dir / "traces"
    method_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{method}] loading engine", flush=True)
    engine = make_engine(args, method)

    prompts = build_prompts(args)
    warmup_prompts = prompts[: max(1, min(args.num_warmup_requests, len(prompts)))]
    warmup_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": 0.0,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": 1.0,
        "max_new_tokens": args.warmup_tokens,
        "ignore_eos": args.ignore_eos,
    }
    profile_params = dict(warmup_params)
    profile_params["max_new_tokens"] = args.max_new_tokens

    print(
        f"[{method}] warmup {len(warmup_prompts)} request(s), max_new_tokens={args.warmup_tokens}",
        flush=True,
    )
    engine.generate(prompt=warmup_prompts, sampling_params=warmup_params, return_logprob=False)
    torch.cuda.synchronize()

    print(
        f"[{method}] start profiler, {len(prompts)} request(s), max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    start = time.perf_counter()
    engine.start_profile(
        output_dir=str(trace_dir),
        activities=["CPU", "GPU"],
        with_stack=False,
        record_shapes=False,
        profile_by_stage=False,
        merge_profiles=True,
        profile_prefix=method,
    )
    output = engine.generate(
        prompt=prompts,
        sampling_params=profile_params,
        return_logprob=False,
    )
    torch.cuda.synchronize()
    engine.stop_profile()
    elapsed = time.perf_counter() - start

    out_tokens = output_token_count(output)
    summary = {
        "method": method,
        "elapsed_sec_including_profile": elapsed,
        "output_tokens": out_tokens,
        "output_tok_s_including_profile": (out_tokens / elapsed if out_tokens else None),
        "max_new_tokens": args.max_new_tokens,
        "warmup_tokens": args.warmup_tokens,
        "num_profile_requests": len(prompts),
        "num_warmup_requests": len(warmup_prompts),
        "trace_dir": str(trace_dir),
        "engine_config": {
            "tp_size": args.tp_size,
            "mem_fraction_static": args.mem_fraction_static,
            "csd_freq_threshold": args.csd_freq_threshold,
            "csd_prob_ratio": args.csd_prob_ratio,
            "max_running_requests": args.max_running_requests,
            "entropy_p20_threshold": ENTROPY_P20_THRESHOLD if method in {"plain_entropy_p20", "p20_no_ratio"} else None,
        },
    }
    (method_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{method}] done: {json.dumps(summary, ensure_ascii=False)}", flush=True)

    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/model/Qwen3.5-35B-A3B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--csd-table-path", required=True)
    parser.add_argument("--csd-freq-threshold", type=int, default=6)
    parser.add_argument("--csd-prob-ratio", type=float, default=0.3)
    parser.add_argument("--csd-rebuild-threshold", type=int, default=4096)
    parser.add_argument("--methods", nargs="+", default=["plain", "dynamic_no_ratio", "p20_no_ratio"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--port", type=int, default=30111)
    parser.add_argument("--mem-fraction-static", type=float, default=0.72)
    parser.add_argument("--watchdog-timeout", type=int, default=7200)
    parser.add_argument("--mamba-scheduler-strategy", default="no_buffer")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-profile-requests", type=int, default=4)
    parser.add_argument("--num-warmup-requests", type=int, default=1)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument(
        "--prompt-indices",
        default=None,
        help="Comma-separated 0-based indices to select from --prompt-file.",
    )
    args = parser.parse_args()

    run_dir = Path(args.out_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts.json").write_text(
        json.dumps(build_prompts(args), indent=2, ensure_ascii=False)
    )

    all_summaries = []
    for idx, method in enumerate(args.methods):
        args.port = args.port + idx
        all_summaries.append(run_method(args, method, run_dir))
    (run_dir / "summary_all.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))
    print(f"all done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
