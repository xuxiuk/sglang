#!/usr/bin/env python3
"""Run lm-eval against SGLang native /generate and save speculative metrics."""

import argparse
import json
import time
from pathlib import Path

import lm_eval
import requests
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table

from lm_eval_sglang_native import SGLangNativeLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lm-eval with SGLang /generate and collect speculative/CSD metrics."
    )
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--model", required=True, help="HF model/tokenizer name or path")
    parser.add_argument("--tasks", required=True, help="Comma-separated lm-eval task names")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--num-concurrent", type=int, default=1)
    parser.add_argument("--max-gen-toks", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-backend", default="huggingface")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--fewshot-as-multiturn", action="store_true")
    parser.add_argument("--gen-kwargs", default=None)
    parser.add_argument("--output-path", default="lm_eval_sglang_native_results.json")
    parser.add_argument(
        "--metrics-output-path", default="lm_eval_sglang_native_metrics.json"
    )
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--csd-table-path", default=None)
    parser.add_argument("--csd-freq-threshold", type=int, default=None)
    parser.add_argument("--csd-prob-ratio", type=float, default=None)
    parser.add_argument("--csd-dynamic-update", action="store_true")
    parser.add_argument("--csd-force-accept-disabled", action="store_true")
    parser.add_argument("--speculative-algorithm", default=None)
    parser.add_argument("--speculative-num-steps", type=int, default=None)
    parser.add_argument("--speculative-eagle-topk", type=int, default=None)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=None)
    return parser.parse_args()


def _get_json(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": repr(exc)}


def _pick_server_config(server_info):
    keys = [
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "speculative_csd",
        "speculative_csd_dynamic_update",
        "speculative_csd_force_accept_disabled",
        "speculative_csd_table_path",
        "speculative_csd_freq_threshold",
        "speculative_csd_prob_ratio",
        "speculative_csd_delta_capacity",
        "tp_size",
        "dp_size",
        "mem_fraction_static",
        "max_running_requests",
        "context_length",
        "version",
    ]
    return {key: server_info.get(key) for key in keys if key in server_info}


def _run_config(args):
    return {
        "run_tag": args.run_tag,
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_concurrent": args.num_concurrent,
        "max_gen_toks": args.max_gen_toks,
        "max_length": args.max_length,
        "tokenizer_backend": args.tokenizer_backend,
        "trust_remote_code": args.trust_remote_code,
        "apply_chat_template": args.apply_chat_template,
        "fewshot_as_multiturn": args.fewshot_as_multiturn,
        "gen_kwargs": args.gen_kwargs,
        "speculative": {
            "algorithm": args.speculative_algorithm,
            "num_steps": args.speculative_num_steps,
            "eagle_topk": args.speculative_eagle_topk,
            "num_draft_tokens": args.speculative_num_draft_tokens,
        },
        "csd": {
            "table_path": args.csd_table_path,
            "freq_threshold": args.csd_freq_threshold,
            "prob_ratio": args.csd_prob_ratio,
            "dynamic_update": args.csd_dynamic_update,
            "force_accept_disabled": args.csd_force_accept_disabled,
        },
    }


def main():
    args = parse_args()

    model = SGLangNativeLM(
        model=args.model,
        base_url=args.base_url,
        tokenizer=args.tokenizer,
        tokenizer_backend=args.tokenizer_backend,
        trust_remote_code=args.trust_remote_code,
        batch_size=args.batch_size,
        num_concurrent=args.num_concurrent,
        max_gen_toks=args.max_gen_toks,
        max_length=args.max_length,
        metrics_output_path=args.metrics_output_path,
    )

    task_manager = TaskManager()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    model_info = _get_json(args.base_url.rstrip("/") + "/model_info")
    server_info = _get_json(args.base_url.rstrip("/") + "/server_info")

    start_time = time.perf_counter()
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
    )
    elapsed = time.perf_counter() - start_time

    spec_summary = model.save_spec_metrics()
    total_completion_tokens = (
        spec_summary.get("total_completion_tokens", 0) if spec_summary else 0
    )
    total_prompt_tokens = spec_summary.get("total_prompt_tokens", 0) if spec_summary else 0
    performance = {
        "elapsed_sec": round(elapsed, 3),
        "request_throughput": round((spec_summary.get("total_requests", 0) if spec_summary else 0) / elapsed, 3)
        if elapsed > 0
        else 0,
        "output_token_throughput": round(total_completion_tokens / elapsed, 3)
        if elapsed > 0
        else 0,
        "total_token_throughput": round((total_prompt_tokens + total_completion_tokens) / elapsed, 3)
        if elapsed > 0
        else 0,
    }

    results.setdefault("sglang", {})
    results["sglang"].update(
        {
            "run_config": _run_config(args),
            "model_info": model_info,
            "server_config": _pick_server_config(server_info),
            "performance": performance,
            "speculative_metrics": spec_summary,
        }
    )

    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))

    output_path = Path(args.output_path)
    output_path.write_text(
        json.dumps(results, indent=2, default=handle_non_serializable),
        encoding="utf-8",
    )
    print(f"lm-eval results saved to {output_path}")


if __name__ == "__main__":
    main()
