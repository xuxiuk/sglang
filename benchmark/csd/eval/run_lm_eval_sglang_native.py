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
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--num-concurrent", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-gen-toks", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-backend", default="huggingface")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument(
        "--fewshot-as-multiturn",
        choices=("auto", "true", "false"),
        default="auto",
    )
    parser.add_argument("--gen-kwargs", default=None)
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument(
        "--sample-log-path",
        default=None,
        help="Write compact per-sample prompts, outputs, filtered answers, and targets to JSONL.",
    )
    parser.add_argument("--output-path", default="lm_eval_sglang_native_results.json")
    parser.add_argument(
        "--metrics-output-path", default="lm_eval_sglang_native_metrics.json"
    )
    parser.add_argument(
        "--result-jsonl-path",
        default=None,
        help="Append a compact benchmark result JSON for this run to the given JSONL file.",
    )
    parser.add_argument("--parallel", default=None)
    parser.add_argument("--repeat-index", type=int, default=None)
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--experiment-config-json", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--watchdog-timeout", type=int, default=None)
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


def _fewshot_as_multiturn_value(args):
    if args.fewshot_as_multiturn == "auto":
        return bool(args.apply_chat_template)
    return args.fewshot_as_multiturn == "true"


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
        "speculative_csd_enabled",
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
        "mode": args.mode,
        "experiment_config": json.loads(args.experiment_config_json)
        if args.experiment_config_json
        else None,
        "server_log": args.server_log,
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "cuda_devices": args.cuda_devices,
        "tensor_parallel_size": args.tensor_parallel_size,
        "mem_fraction_static": args.mem_fraction_static,
        "watchdog_timeout": args.watchdog_timeout,
        "parallel": args.parallel,
        "repeat_index": args.repeat_index,
        "sweep_id": args.sweep_id,
        "tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_concurrent": args.num_concurrent,
        "timeout": args.timeout,
        "max_gen_toks": args.max_gen_toks,
        "max_length": args.max_length,
        "tokenizer_backend": args.tokenizer_backend,
        "trust_remote_code": args.trust_remote_code,
        "apply_chat_template": args.apply_chat_template,
        "fewshot_as_multiturn": _fewshot_as_multiturn_value(args),
        "fewshot_as_multiturn_arg": args.fewshot_as_multiturn,
        "gen_kwargs": args.gen_kwargs,
        "confirm_run_unsafe_code": args.confirm_run_unsafe_code,
        "include_samples": args.include_samples,
        "sample_log_path": args.sample_log_path,
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


def _primary_score(task_metrics):
    for key, value in task_metrics.items():
        if key.endswith("_stderr") or "_stderr," in key:
            continue
        if isinstance(value, (int, float)):
            return key, value
    return None, None


def _compact_result_rows(results, args):
    run_config = results["sglang"]["run_config"]
    performance = results["sglang"]["performance"]
    spec = results["sglang"].get("speculative_metrics") or {}
    total_requests = spec.get("total_requests") or 0
    avg_prompt_tokens = (
        round(spec.get("total_prompt_tokens", 0) / total_requests, 3)
        if total_requests
        else None
    )
    avg_completion_tokens = (
        round(spec.get("total_completion_tokens", 0) / total_requests, 3)
        if total_requests
        else None
    )
    server_config = results["sglang"].get("server_config") or {}
    csd_enabled = server_config.get("speculative_csd_enabled")
    csd_config = {**run_config["csd"], "enabled": csd_enabled}
    task_names = run_config["tasks"]
    n_samples = results.get("n-samples", {})
    n_shot = results.get("n-shot", {})
    experiment_config = run_config.get("experiment_config") or {}
    parallel = args.parallel or experiment_config.get("current_parallel")
    repeat_index = args.repeat_index if args.repeat_index is not None else experiment_config.get("repeat_index")
    sweep_id = args.sweep_id or experiment_config.get("sweep_id")
    rows = []
    for task_name in task_names:
        task_metrics = results.get("results", {}).get(task_name, {})
        score_key, score_value = _primary_score(task_metrics)
        num_requests = None
        if isinstance(n_samples.get(task_name), dict):
            num_requests = n_samples[task_name].get("effective")
        row = {
            "parallel": parallel,
            "resolved_num_concurrent": args.num_concurrent,
            "repeat_index": repeat_index,
            "sweep_id": sweep_id,
            "task": task_name,
            "backend": "srt",
            "num_gpus": args.tensor_parallel_size,
            "latency": performance.get("elapsed_sec"),
            "accuracy": round(score_value, 6) if isinstance(score_value, float) else score_value,
            "invalid": None,
            "throughput": performance.get("output_token_throughput"),
            "accept_length": spec.get("avg_spec_accept_length"),
            "aggregate_accept_length": spec.get("aggregate_spec_accept_length"),
            "spec_success_rate": spec.get("spec_success_rate"),
            "avg_spec_success_rate": spec.get("avg_spec_success_rate"),
            "aggregate_spec_accept_rate": spec.get("aggregate_spec_accept_rate"),
            "spec_output_token_saved_ratio": spec.get("spec_output_token_saved_ratio"),
            "spec_verify_ct": spec.get("total_spec_verify_ct"),
            "speculative_num_steps": spec.get("speculative_num_steps"),
            "spec_draft_token_num": spec.get("total_spec_draft_token_num"),
            "csd_enabled": csd_enabled,
            "csd_lookup_hit_ct": spec.get("total_csd_lookup_hits"),
            "csd_forced_accept_ct": spec.get("total_csd_forced_accepts"),
            "csd_delta_pair_ct": spec.get("total_csd_delta_pairs"),
            "num_requests": num_requests,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "other": {
                "parallel": parallel,
                "resolved_num_concurrent": args.num_concurrent,
                "repeat_index": repeat_index,
                "sweep_id": sweep_id,
                "task": task_name,
                "score_key": score_key,
                "metrics": task_metrics,
                "num_fewshot": n_shot.get(task_name),
                "limit": args.limit,
                "batch_size": args.batch_size,
                "num_concurrent": args.num_concurrent,
                "max_gen_toks": args.max_gen_toks,
                "max_length": args.max_length,
                "avg_prompt_tokens": avg_prompt_tokens,
                "avg_completion_tokens": avg_completion_tokens,
                "apply_chat_template": args.apply_chat_template,
                "fewshot_as_multiturn": _fewshot_as_multiturn_value(args),
                "backend": "srt",
                "host": args.base_url,
                "model": args.model,
                "tokenizer": args.tokenizer,
                "run_tag": args.run_tag,
                "mode": args.mode,
                "server_log": args.server_log,
                "speculative": run_config["speculative"],
                "csd_enabled": csd_enabled,
                "csd": csd_config,
                "performance": performance,
                "speculative_metrics": {
                    key: value
                    for key, value in spec.items()
                    if key != "per_request_metrics"
                },
                "server_config": results["sglang"].get("server_config"),
            },
        }
        rows.append(row)
    return rows


def _write_sample_log(results, args):
    if not args.sample_log_path:
        return
    samples = results.get("samples") or {}
    sample_log_path = Path(args.sample_log_path)
    with sample_log_path.open("a", encoding="utf-8") as f:
        for task_name, rows in samples.items():
            for row in rows:
                arguments = row.get("arguments") or {}
                prompt = None
                gen_args = arguments.get("gen_args_0") if isinstance(arguments, dict) else None
                if isinstance(gen_args, dict):
                    prompt = gen_args.get("arg_0")
                record = {
                    "task": task_name,
                    "doc_id": row.get("doc_id"),
                    "target": row.get("target"),
                    "prompt": prompt,
                    "resps": row.get("resps"),
                    "filtered_resps": row.get("filtered_resps"),
                    "exact_match": row.get("exact_match"),
                    "filter": row.get("filter"),
                    "run_tag": args.run_tag,
                    "mode": args.mode,
                    "apply_chat_template": args.apply_chat_template,
                    "fewshot_as_multiturn": _fewshot_as_multiturn_value(args),
                    "max_gen_toks": args.max_gen_toks,
                    "gen_kwargs": args.gen_kwargs,
                }
                f.write(json.dumps(record, default=handle_non_serializable) + "\n")
    print(f"lm-eval sample log appended to {sample_log_path}")


def main():
    args = parse_args()

    model_kwargs = {
        "model": args.model,
        "base_url": args.base_url,
        "tokenizer": args.tokenizer,
        "tokenizer_backend": args.tokenizer_backend,
        "trust_remote_code": args.trust_remote_code,
        "batch_size": args.batch_size,
        "timeout": args.timeout,
        "max_gen_toks": args.max_gen_toks,
        "max_length": args.max_length,
        "metrics_output_path": args.metrics_output_path,
        "speculative_num_steps": args.speculative_num_steps,
    }
    if args.num_concurrent is not None:
        model_kwargs["num_concurrent"] = args.num_concurrent

    model = SGLangNativeLM(**model_kwargs)

    task_manager = TaskManager()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    model_info = _get_json(args.base_url.rstrip("/") + "/model_info")
    server_info = _get_json(args.base_url.rstrip("/") + "/server_info")

    start_time = time.perf_counter()
    fewshot_as_multiturn = _fewshot_as_multiturn_value(args)
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        log_samples=args.include_samples or bool(args.sample_log_path),
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
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

    server_config = _pick_server_config(server_info)
    run_config = _run_config(args)
    run_config["csd"]["enabled"] = server_config.get("speculative_csd_enabled")

    results.setdefault("sglang", {})
    results["sglang"].update(
        {
            "run_config": run_config,
            "model_info": model_info,
            "server_config": server_config,
            "performance": performance,
            "speculative_metrics": spec_summary,
        }
    )
    if args.sample_log_path:
        _write_sample_log(results, args)
    if not args.include_samples:
        results.pop("samples", None)

    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))

    output_path = Path(args.output_path)
    output_path.write_text(
        json.dumps(results, indent=2, default=handle_non_serializable),
        encoding="utf-8",
    )
    print(f"lm-eval results saved to {output_path}")

    if args.result_jsonl_path:
        result_jsonl_path = Path(args.result_jsonl_path)
        with result_jsonl_path.open("a", encoding="utf-8") as f:
            for row in _compact_result_rows(results, args):
                f.write(json.dumps(row, default=handle_non_serializable) + "\n")
        print(f"lm-eval compact result appended to {result_jsonl_path}")


if __name__ == "__main__":
    main()
