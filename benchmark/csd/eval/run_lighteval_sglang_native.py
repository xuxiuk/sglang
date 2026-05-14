#!/usr/bin/env python3
"""Run LightEval with SGLang Engine and save speculative/CSD metrics."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

CSD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SGLANG_SRC = Path(os.environ.get("SGLANG_SRC", REPO_ROOT / "python"))
LIGHTEVAL_SRC = Path(os.environ.get("LIGHTEVAL_SRC", CSD_ROOT / "lighteval" / "src"))
for src_path in (SGLANG_SRC, LIGHTEVAL_SRC):
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from lighteval.logging.evaluation_tracker import EnhancedJSONEncoder, EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.sglang.sglang_model import SGLangModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


PREFERRED_SCORE_KEYS = (
    "exact_match",
    "expr_gold_metric",
    "pass@1",
    "pass_at_1",
    "avg@1",
    "avg_at_1",
    "acc",
    "accuracy",
    "qem",
    "f1",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LightEval with SGLang Engine and collect speculative/CSD metrics."
    )
    parser.add_argument("--model", required=True, help="HF model path/name")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer path/name")
    parser.add_argument("--tasks", required=True, help="Comma-separated LightEval task specs, e.g. gsm8k|0")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per task")
    parser.add_argument("--output-dir", default=None, help="LightEval tracker output dir")
    parser.add_argument("--output-path", default="lighteval_sglang_native_results.json")
    parser.add_argument("--metrics-output-path", default="lighteval_sglang_native_metrics.json")
    parser.add_argument("--result-jsonl-path", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--experiment-config-json", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--watchdog-timeout", type=int, default=None)
    parser.add_argument("--mamba-scheduler-strategy", default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--max-gen-toks", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--add-special-tokens", action="store_true", default=True)
    parser.add_argument("--no-add-special-tokens", dest="add_special_tokens", action="store_false")
    parser.add_argument("--pairwise-tokenization", action="store_true")
    parser.add_argument("--sampling-backend", default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--override-chat-template", choices=("true", "false", "auto"), default="auto")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--gen-kwargs", default=None)
    parser.add_argument("--save-details", action="store_true")
    parser.add_argument("--disable-sample-cache", action="store_true")
    parser.add_argument("--dataset-loading-processes", type=int, default=1)
    parser.add_argument("--custom-tasks", default=None)
    parser.add_argument("--num-fewshot-seeds", type=int, default=1)
    parser.add_argument("--remove-reasoning-tags", action="store_true", default=True)
    parser.add_argument("--keep-reasoning-tags", dest="remove_reasoning_tags", action="store_false")
    parser.add_argument("--reasoning-tags", default="[('<think>', '</think>')]")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--speculative-algorithm", default=None)
    parser.add_argument("--speculative-num-steps", type=int, default=None)
    parser.add_argument("--speculative-eagle-topk", type=int, default=None)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=None)
    parser.add_argument("--csd-table-path", default=None)
    parser.add_argument("--csd-save-table-path", default=None)
    parser.add_argument("--csd-freq-threshold", type=int, default=None)
    parser.add_argument("--csd-prob-ratio", type=float, default=None)
    parser.add_argument("--csd-dynamic-update", action="store_true")
    parser.add_argument("--csd-force-accept-disabled", action="store_true")
    parser.add_argument("--csd-enabled", action="store_true")
    return parser.parse_args()


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in ("1", "true", "yes", "y"):
        return True
    if lowered in ("0", "false", "no", "n"):
        return False
    return value


def _parse_gen_kwargs(gen_kwargs):
    params = {"max_new_tokens": None, "temperature": 0}
    if not gen_kwargs:
        return params
    for item in gen_kwargs.split(","):
        if not item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = _parse_bool(value)
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        if key == "seed":
            key = "sampling_seed"
        params[key] = value
    return params


def _run_config(args):
    return {
        "run_tag": args.run_tag,
        "mode": args.mode,
        "experiment_config": json.loads(args.experiment_config_json) if args.experiment_config_json else None,
        "server_log": args.server_log,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "cuda_devices": args.cuda_devices,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
        "watchdog_timeout": args.watchdog_timeout,
        "mamba_scheduler_strategy": args.mamba_scheduler_strategy,
        "log_level": args.log_level,
        "tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
        "limit": args.limit,
        "max_gen_toks": args.max_gen_toks,
        "max_length": args.max_length,
        "trust_remote_code": args.trust_remote_code,
        "gen_kwargs": args.gen_kwargs,
        "disable_sample_cache": args.disable_sample_cache,
        "eval_framework": "lighteval",
        "speculative": {
            "algorithm": args.speculative_algorithm,
            "num_steps": args.speculative_num_steps,
            "eagle_topk": args.speculative_eagle_topk,
            "num_draft_tokens": args.speculative_num_draft_tokens,
        },
        "csd": {
            "enabled": args.csd_enabled,
            "table_path": args.csd_table_path,
            "save_table_path": args.csd_save_table_path,
            "freq_threshold": args.csd_freq_threshold,
            "prob_ratio": args.csd_prob_ratio,
            "dynamic_update": args.csd_dynamic_update,
            "force_accept_disabled": args.csd_force_accept_disabled,
        },
    }


def _server_config(args):
    return {
        "speculative_algorithm": args.speculative_algorithm,
        "speculative_num_steps": args.speculative_num_steps,
        "speculative_eagle_topk": args.speculative_eagle_topk,
        "speculative_num_draft_tokens": args.speculative_num_draft_tokens,
        "speculative_csd_enabled": args.csd_enabled,
        "speculative_csd_dynamic_update": args.csd_dynamic_update,
        "speculative_csd_force_accept_disabled": args.csd_force_accept_disabled,
        "speculative_csd_table_path": args.csd_table_path,
        "speculative_csd_save_table_path": args.csd_save_table_path,
        "speculative_csd_freq_threshold": args.csd_freq_threshold,
        "speculative_csd_prob_ratio": args.csd_prob_ratio,
        "tp_size": args.tensor_parallel_size,
        "dp_size": args.data_parallel_size,
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
        "context_length": args.max_length,
        "watchdog_timeout": args.watchdog_timeout,
        "mamba_scheduler_strategy": args.mamba_scheduler_strategy,
        "log_level": args.log_level,
    }


def _primary_score(task_metrics):
    if not isinstance(task_metrics, dict):
        return None, None
    for key in PREFERRED_SCORE_KEYS:
        value = task_metrics.get(key)
        if isinstance(value, (int, float)):
            return key, value
    for key, value in task_metrics.items():
        if key.endswith("_stderr") or "stderr" in key or key.startswith("alias"):
            continue
        if isinstance(value, (int, float)):
            return key, value
    return None, None


def _task_base_name(task_name):
    parts = task_name.split("|")
    if len(parts) == 3:
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    base, _, fewshot = task_name.rpartition(":")
    if base and fewshot.isdigit():
        return base
    return task_name


def _find_task_key(mapping, task_name):
    if task_name in mapping:
        return task_name
    base_name = _task_base_name(task_name)
    for key in mapping:
        if key == "all" or ":_average" in key:
            continue
        if _task_base_name(key) == base_name:
            return key
    return None


def _task_metrics(results, task_name):
    metrics = results.get("results", {})
    task_key = _find_task_key(metrics, task_name)
    return metrics.get(task_key, {}) if task_key else {}


def _num_requests(results, task_name):
    summary = results.get("summary_tasks", {})
    task_key = _find_task_key(summary, task_name)
    item = summary.get(task_key) if task_key else None
    if isinstance(item, dict):
        count = item.get("truncated", 0) + item.get("non_truncated", 0)
        if count:
            return count
    spec = results.get("sglang", {}).get("speculative_metrics") or {}
    return spec.get("total_requests")


def _compact_result_rows(results, args):
    run_config = results["sglang"]["run_config"]
    performance = results["sglang"]["performance"]
    spec = results["sglang"].get("speculative_metrics") or {}
    total_requests = spec.get("total_requests") or 0
    avg_prompt_tokens = round(spec.get("total_prompt_tokens", 0) / total_requests, 3) if total_requests else None
    avg_completion_tokens = (
        round(spec.get("total_completion_tokens", 0) / total_requests, 3) if total_requests else None
    )
    server_config = results["sglang"].get("server_config") or {}
    csd_enabled = server_config.get("speculative_csd_enabled")
    csd_config = {**run_config["csd"], "enabled": csd_enabled}
    rows = []
    for task_name in run_config["tasks"]:
        metrics = _task_metrics(results, task_name)
        score_key, score_value = _primary_score(metrics)
        row = {
            "task": task_name,
            "backend": "srt",
            "eval_framework": "lighteval",
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
            "num_requests": _num_requests(results, task_name),
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "other": {
                "task": task_name,
                "score_key": score_key,
                "metrics": metrics,
                "limit": args.limit,
                "max_running_requests": args.max_running_requests,
                "max_gen_toks": args.max_gen_toks,
                "max_length": args.max_length,
                "avg_prompt_tokens": avg_prompt_tokens,
                "avg_completion_tokens": avg_completion_tokens,
                "backend": "srt",
                "eval_framework": "lighteval",
                "model": args.model,
                "tokenizer": args.tokenizer,
                "run_tag": args.run_tag,
                "mode": args.mode,
                "server_log": args.server_log,
                "speculative": run_config["speculative"],
                "csd_enabled": csd_enabled,
                "csd": csd_config,
                "performance": performance,
                "speculative_metrics": {key: value for key, value in spec.items() if key != "per_request_metrics"},
                "server_config": server_config,
            },
        }
        rows.append(row)
    return rows


def _override_chat_template(value):
    if value == "auto":
        return None
    return value == "true"


def _build_model_config(args):
    gen_params = _parse_gen_kwargs(args.gen_kwargs)
    gen_params["max_new_tokens"] = args.max_gen_toks
    generation_parameters = GenerationParameters(**gen_params)
    return SGLangModelConfig(
        model_name=args.model,
        tokenizer_path=args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        load_format=args.load_format,
        dtype=args.dtype,
        tp_size=args.tensor_parallel_size,
        dp_size=args.data_parallel_size,
        context_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        add_special_tokens=args.add_special_tokens,
        pairwise_tokenization=args.pairwise_tokenization,
        sampling_backend=args.sampling_backend,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        chunked_prefill_size=args.chunked_prefill_size,
        watchdog_timeout=args.watchdog_timeout,
        mamba_scheduler_strategy=args.mamba_scheduler_strategy,
        log_level=args.log_level,
        override_chat_template=_override_chat_template(args.override_chat_template),
        system_prompt=args.system_prompt,
        generation_parameters=generation_parameters,
        use_sample_cache=not args.disable_sample_cache,
        speculative_algorithm=args.speculative_algorithm,
        speculative_num_steps=args.speculative_num_steps,
        speculative_eagle_topk=args.speculative_eagle_topk,
        speculative_num_draft_tokens=args.speculative_num_draft_tokens,
        speculative_csd_enabled=args.csd_enabled,
        speculative_csd_table_path=args.csd_table_path if args.csd_enabled else None,
        speculative_csd_freq_threshold=args.csd_freq_threshold if args.csd_enabled else None,
        speculative_csd_prob_ratio=args.csd_prob_ratio if args.csd_enabled else None,
        speculative_csd_dynamic_update=args.csd_dynamic_update,
        speculative_csd_force_accept_disabled=args.csd_force_accept_disabled,
        speculative_csd_save_table_path=args.csd_save_table_path,
        speculative_csd_save_table_metadata=_run_config(args) if args.csd_save_table_path else None,
    )


def main():
    args = parse_args()
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    output_path = Path(args.output_path)
    metrics_path = Path(args.metrics_output_path)
    if args.csd_save_table_path and Path(args.csd_save_table_path).exists():
        raise FileExistsError(f"Refusing to overwrite existing CSD table: {args.csd_save_table_path}")
    if args.csd_save_table_path:
        Path(args.csd_save_table_path).parent.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir or str(output_path.parent / "lighteval_tracker")

    model_config = _build_model_config(args)
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=args.save_details,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.SGLANG,
        dataset_loading_processes=args.dataset_loading_processes,
        custom_tasks_directory=args.custom_tasks,
        num_fewshot_seeds=args.num_fewshot_seeds,
        max_samples=args.limit,
        remove_reasoning_tags=args.remove_reasoning_tags,
        reasoning_tags=args.reasoning_tags,
        bootstrap_iters=args.bootstrap_iters,
    )
    pipeline = Pipeline(
        tasks=args.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    start_time = time.perf_counter()
    pipeline.evaluate()
    elapsed = time.perf_counter() - start_time
    results = pipeline.get_results()

    spec_summary = pipeline.model.get_spec_metrics()
    if spec_summary is not None:
        metrics_path.write_text(json.dumps(spec_summary, indent=2), encoding="utf-8")
        print(f"LightEval SGLang metrics saved to {metrics_path}")

    total_requests = spec_summary.get("total_requests", 0) if spec_summary else 0
    total_prompt_tokens = spec_summary.get("total_prompt_tokens", 0) if spec_summary else 0
    total_completion_tokens = spec_summary.get("total_completion_tokens", 0) if spec_summary else 0
    performance = {
        "elapsed_sec": round(elapsed, 3),
        "request_throughput": round(total_requests / elapsed, 3) if elapsed > 0 else 0,
        "output_token_throughput": round(total_completion_tokens / elapsed, 3) if elapsed > 0 else 0,
        "total_token_throughput": round((total_prompt_tokens + total_completion_tokens) / elapsed, 3)
        if elapsed > 0
        else 0,
    }

    run_config = _run_config(args)
    server_config = _server_config(args)
    results.setdefault("sglang", {})
    results["sglang"].update(
        {
            "run_config": run_config,
            "model_info": {"model_path": args.model, "tokenizer_path": args.tokenizer or args.model},
            "server_config": server_config,
            "performance": performance,
            "speculative_metrics": spec_summary,
        }
    )

    output_path.write_text(
        json.dumps(results, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"LightEval results saved to {output_path}")

    if args.result_jsonl_path:
        result_jsonl_path = Path(args.result_jsonl_path)
        with result_jsonl_path.open("a", encoding="utf-8") as f:
            for row in _compact_result_rows(results, args):
                f.write(json.dumps(row, cls=EnhancedJSONEncoder, ensure_ascii=False) + "\n")
        print(f"LightEval compact result appended to {result_jsonl_path}")


if __name__ == "__main__":
    main()
