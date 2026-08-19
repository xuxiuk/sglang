#!/usr/bin/env python3
"""Evaluate an already-running DSpark server on long reasoning/code tasks.

This is an HTTP-only client.  It deliberately disables both LightEval's sample
cache and LiteLLM's response cache so different CSD/entropy server modes cannot
silently reuse one another's generations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=(
            "aime25",
            "aime25_avg4",
            "aime25_avg16",
            "math500_avg4",
            "lcb",
            "gsm8k_avg4",
        ),
        required=True,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="/data/model/DeepSeek-V4-Flash-DSpark")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--parallel", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=81920)
    parser.add_argument("--max-model-length", type=int, default=96000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reasoning-effort", choices=("high", "max"), default="high"
    )
    parser.add_argument("--timeout", type=float, default=86400)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retries after the initial HTTP attempt. Formal runs use zero.",
    )
    parser.add_argument(
        "--independent-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Split an n>1 metric request into independent n=1 HTTP requests, "
            "then combine the choices for LightEval metric aggregation."
        ),
    )
    parser.add_argument(
        "--request-timing-file",
        type=Path,
        help=(
            "Save non-streaming per-request start/end time and token usage as "
            "JSONL. Aggregate output throughput is computed over the union of "
            "request wall-time, so concurrent requests do not double-count time."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import litellm
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    # Select the frozen task definitions before Registry constructs the task.
    # The avg4 variants request exactly four independent generations per item.
    system_prompt = None
    if args.task == "lcb":
        from lighteval.tasks.tasks.lcb import main as lcb_tasks

        for task_config in lcb_tasks.TASKS_TABLE:
            if task_config.name == "lcb:codegeneration_v6":
                task_config.generation_size = args.max_new_tokens
        task_spec = "lcb:codegeneration_v6|0"
        system_prompt = lcb_tasks.SYSTEM_MESSAGE_GENERIC
    elif args.task in {"aime25_avg4", "aime25_avg16"}:
        from lighteval.metrics.metrics import Metrics
        from lighteval.tasks.tasks import aime as aime_tasks

        for task_config in aime_tasks.TASKS_TABLE:
            if task_config.name == "aime25_avg":
                task_config.generation_size = args.max_new_tokens
                n = 16 if args.task == "aime25_avg16" else 4
                task_config.metrics = [
                    Metrics.avg_at_n_math(sample_params={"n": n}),
                    Metrics.pass_at_k_math(sample_params={"k": n, "n": n}),
                ]
        task_spec = "aime25_avg|0"
    elif args.task == "math500_avg4":
        from lighteval.metrics.metrics import Metrics
        from lighteval.tasks.tasks import math_500 as math500_tasks

        for task_config in math500_tasks.TASKS_TABLE:
            if task_config.name == "math_500":
                task_config.generation_size = args.max_new_tokens
                task_config.metrics = [
                    Metrics.avg_at_n_math(sample_params={"n": 4}),
                    Metrics.pass_at_k_math(sample_params={"k": 4, "n": 4}),
                ]
        task_spec = "math_500|0"
    elif args.task == "gsm8k_avg4":
        from lighteval.metrics.metrics import Metrics
        from lighteval.tasks.tasks import gsm8k as gsm8k_tasks

        for task_config in gsm8k_tasks.TASKS_TABLE:
            if task_config.name == "gsm8k_avg":
                task_config.generation_size = args.max_new_tokens
                task_config.metrics = [
                    Metrics.avg_at_n_math(sample_params={"n": 4}),
                    Metrics.pass_at_k_math(sample_params={"k": 4, "n": 4}),
                ]
        task_spec = "gsm8k_avg|0"
    else:
        task_spec = "aime25|0"

    # LiteLLM only exposes OpenAI-standard sampling knobs directly.  SGLang
    # accepts top_k/min_p/repetition_penalty in extra_body, while
    # presence_penalty is an OpenAI-standard field.
    from lighteval.models.model_input import GenerationParameters

    def to_sglang_litellm_dict(self: GenerationParameters) -> dict:
        direct = {
            "max_completion_tokens": self.max_new_tokens,
            "stop": self.stop_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        extra = {
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }
        result = {key: value for key, value in direct.items() if value is not None}
        extra = {key: value for key, value in extra.items() if value is not None}
        if extra:
            result["extra_body"] = extra
        return result

    GenerationParameters.to_litellm_dict = to_sglang_litellm_dict

    # The stock endpoint backend hard-codes caching=True.  Force it off: cache
    # reuse across entropy thresholds would invalidate the comparison.
    original_completion = litellm.completion
    timing_records: list[dict] = []
    timing_lock = threading.Lock()
    sample_context = threading.local()
    request_counter = 0

    def record_timing(
        *, messages, started_at, ended_at, response, sample_index=None
    ) -> None:
        nonlocal request_counter
        usage = getattr(response, "usage", None)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        choices = list(getattr(response, "choices", []) or [])
        prompt_payload = json.dumps(
            messages, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        with timing_lock:
            request_counter += 1
            record = {
                "request_id": request_counter,
                "prompt_sha256": hashlib.sha256(prompt_payload).hexdigest(),
                "sample_index": sample_index,
                "request_start_s": started_at,
                "request_end_s": ended_at,
                "e2e_s": ended_at - started_at,
                "completion_tokens": completion_tokens,
                "choice_count": len(choices),
                "finish_reasons": [
                    getattr(choice, "finish_reason", None) for choice in choices
                ],
                "streaming": False,
            }
            timing_records.append(record)
            with args.request_timing_file.open("a", encoding="utf-8") as stream_file:
                stream_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream_file.flush()

    def call_one(completion_args, completion_kwargs, sample_index=None):
        if sample_index is None:
            sample_index = getattr(sample_context, "sample_index", None)
        messages = completion_kwargs.get("messages")
        started_at = time.perf_counter()
        response = original_completion(*completion_args, **completion_kwargs)
        ended_at = time.perf_counter()
        if args.request_timing_file is not None:
            record_timing(
                messages=messages,
                started_at=started_at,
                ended_at=ended_at,
                response=response,
                sample_index=sample_index,
            )
        return response

    def combine_independent_responses(responses):
        combined = responses[0]
        choices = []
        completion_tokens = prompt_tokens = total_tokens = 0
        for sample_index, response in enumerate(responses):
            response_choices = list(getattr(response, "choices", []) or [])
            if len(response_choices) != 1:
                raise RuntimeError(
                    "Independent n=1 request returned "
                    f"{len(response_choices)} choices instead of one"
                )
            response_choices[0].index = sample_index
            choices.extend(response_choices)
            usage = getattr(response, "usage", None)
            completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        combined.choices = choices
        usage = getattr(combined, "usage", None)
        if usage is not None:
            usage.completion_tokens = completion_tokens
            usage.prompt_tokens = prompt_tokens
            usage.total_tokens = total_tokens
        return combined

    def completion_without_cache(*completion_args, **completion_kwargs):
        completion_kwargs["caching"] = False
        # LiteLLM has its own retry layer in addition to LightEval's retry
        # loop. Disable it so a timed-out long generation is never resubmitted.
        completion_kwargs["num_retries"] = 0
        # DeepSeek-V4 has no tokenizer_config.json Jinja template.  SGLang's
        # native dsv4 encoder consumes these explicit request settings and
        # renders the release encoding_dsv4 format.
        extra_body = copy.deepcopy(completion_kwargs.get("extra_body") or {})
        chat_template_kwargs = copy.deepcopy(
            extra_body.get("chat_template_kwargs") or {}
        )
        chat_template_kwargs["thinking"] = args.thinking
        chat_template_kwargs["reasoning_effort"] = args.reasoning_effort
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        completion_kwargs["extra_body"] = extra_body
        completion_kwargs["stream"] = False
        return call_one(completion_args, completion_kwargs)

    litellm.completion = completion_without_cache
    litellm.cache = None

    generation = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    config = LiteLLMModelConfig(
        model_name=f"openai/{args.model}",
        provider="openai",
        base_url=args.base_url,
        api_key="EMPTY",
        system_prompt=system_prompt,
        concurrent_requests=args.parallel,
        max_model_length=args.max_model_length,
        # This LightEval version interprets api_max_retry as the total number
        # of attempts (range(api_max_retry)), not retries after the first one.
        api_max_retry=args.max_retries + 1,
        timeout=args.timeout,
        generation_parameters=generation,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.request_timing_file is not None:
        args.request_timing_file.parent.mkdir(parents=True, exist_ok=True)
        if args.request_timing_file.exists():
            raise FileExistsError(
                f"Refusing to overwrite timing file: {args.request_timing_file}"
            )
        args.request_timing_file.touch()
    tracker = EvaluationTracker(
        output_dir=str(args.output_dir / "tracker"),
        save_details=True,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
    )
    pipeline = Pipeline(
        tasks=task_spec,
        pipeline_parameters=PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            dataset_loading_processes=1,
            num_fewshot_seeds=1,
            max_samples=args.max_samples,
            remove_reasoning_tags=True,
            reasoning_tags="[('<think>', '</think>')]",
        ),
        evaluation_tracker=tracker,
        model_config=config,
    )

    if args.independent_samples:

        def independent_call_api_parallel(
            model_self,
            prompts,
            return_logits,
            max_new_tokens,
            num_samples,
            stop_sequence=None,
        ):
            print(
                f"Independent n=1 global queue: problems={len(prompts)}, "
                f"http_concurrency={args.parallel}",
                flush=True,
            )
            def expand(value, count):
                return value if isinstance(value, list) else [value] * count

            count = len(prompts)
            return_logitss = expand(return_logits, count)
            max_new_tokenss = expand(max_new_tokens, count)
            num_sampless = expand(num_samples, count)
            stop_sequencess = [stop_sequence] * count
            jobs = []
            for problem_index, fields in enumerate(
                zip(
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                )
            ):
                prompt, use_logits, max_tokens, samples, stops = fields
                for sample_index in range(samples):
                    jobs.append(
                        (
                            problem_index,
                            sample_index,
                            prompt,
                            use_logits,
                            max_tokens,
                            stops,
                        )
                    )

            def execute(job):
                problem_index, sample_index, prompt, use_logits, max_tokens, stops = job
                sample_context.sample_index = sample_index
                try:
                    response = getattr(
                        model_self, "_LiteLLMClient__call_api"
                    )(prompt, use_logits, max_tokens, 1, stops)
                finally:
                    sample_context.sample_index = None
                return problem_index, sample_index, response

            grouped = [[] for _ in prompts]
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                for problem_index, sample_index, response in executor.map(execute, jobs):
                    grouped[problem_index].append((sample_index, response))

            combined = []
            for problem_responses in grouped:
                ordered = [
                    response
                    for _, response in sorted(
                        problem_responses, key=lambda item: item[0]
                    )
                ]
                combined.append(combine_independent_responses(ordered))
            return combined

        # Both methods use double-underscore names in LiteLLMClient, so assign
        # the class-mangled attribute explicitly.
        setattr(
            type(pipeline.model),
            "_LiteLLMClient__call_api_parallel",
            independent_call_api_parallel,
        )

    # Fail before the expensive generation if an environment accidentally
    # imports an unpatched task definition that falls back to n=1 or n=16.
    metric_names: list[str] = []
    for task in pipeline.tasks_dict.values():
        for metric in task.metrics:
            name = metric.metric_name
            if isinstance(name, list):
                metric_names.extend(name)
            else:
                metric_names.append(name)
    expected_metrics = {
        "aime25_avg4": {"avg@n:n=4", "pass@k:k=4&n=4"},
        "aime25_avg16": {"avg@n:n=16", "pass@k:k=16&n=16"},
        "math500_avg4": {"avg@n:n=4", "pass@k:k=4&n=4"},
        "gsm8k_avg4": {"avg@n:n=4", "pass@k:k=4&n=4"},
        "lcb": {"codegen_pass@1:avg4", "codegen_pass@4"},
    }[args.task]
    if not expected_metrics.issubset(set(metric_names)):
        raise RuntimeError(
            f"Incorrect metrics for {args.task}: expected {sorted(expected_metrics)}, "
            f"got {sorted(metric_names)}"
        )

    # Reinforce the cap on every expanded request. The LiteLLM endpoint backend
    # reads Doc.generation_size when it builds each API call.
    for task in pipeline.tasks_dict.values():
        task.generation_size = args.max_new_tokens
    for docs in pipeline.documents_dict.values():
        for doc in docs:
            doc.generation_size = args.max_new_tokens
    effective_generation_sizes = sorted(
        {doc.generation_size for docs in pipeline.documents_dict.values() for doc in docs}
    )
    if effective_generation_sizes != [args.max_new_tokens]:
        raise RuntimeError(
            "Generation-size override failed: "
            f"expected {args.max_new_tokens}, got {effective_generation_sizes}"
        )
    # Disable the second, LightEval-level sample cache as well.
    pipeline.model._cache = None

    started = time.perf_counter()
    pipeline.evaluate()
    elapsed = time.perf_counter() - started
    results = pipeline.get_results()
    if args.task in {
        "aime25_avg4",
        "aime25_avg16",
        "math500_avg4",
        "gsm8k_avg4",
        "lcb",
    }:
        expected_timing_choices = 1 if args.independent_samples else 4
        bad_choice_counts = [
            row["choice_count"]
            for row in timing_records
            if row["choice_count"] != expected_timing_choices
        ]
        if bad_choice_counts:
            raise RuntimeError(
                "n=4 evaluation returned an unexpected per-request choice count "
                f"(expected {expected_timing_choices}): "
                f"{bad_choice_counts[:10]}"
            )
    pipeline.show_results()
    pipeline.save_and_push_results()

    timing_summary = None
    if args.request_timing_file is not None:
        ordered_records = sorted(timing_records, key=lambda row: row["request_id"])
        timed_records = [
            row
            for row in ordered_records
            if row["completion_tokens"] > 0
            and row["request_end_s"] > row["request_start_s"]
        ]
        completion_tokens = sum(row["completion_tokens"] for row in timed_records)
        per_sample_tokens = sorted(row["completion_tokens"] for row in timed_records)

        def percentile(values: list[int], fraction: float) -> int | None:
            if not values:
                return None
            index = round((len(values) - 1) * fraction)
            return values[index]

        length_finish_count = sum(
            "length" in row.get("finish_reasons", []) for row in timed_records
        )
        intervals = sorted(
            (row["request_start_s"], row["request_end_s"]) for row in timed_records
        )
        active_request_union_seconds = 0.0
        if intervals:
            union_start, union_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= union_end:
                    union_end = max(union_end, end)
                else:
                    active_request_union_seconds += union_end - union_start
                    union_start, union_end = start, end
            active_request_union_seconds += union_end - union_start

        timing_summary = {
            "streaming": False,
            "includes_server_scheduling_and_ttft": True,
            "excludes_client_idle_outside_active_request_intervals": True,
            "request_count": len(ordered_records),
            "logical_problem_count": (
                len(ordered_records)
                // (16 if args.task == "aime25_avg16" else 4)
                if args.independent_samples
                else len(ordered_records)
            ),
            "independent_sample_requests": args.independent_samples,
            "timed_request_count": len(timed_records),
            "completion_tokens": completion_tokens,
            "per_sample_completion_tokens": {
                "mean": (
                    completion_tokens / len(per_sample_tokens)
                    if per_sample_tokens
                    else None
                ),
                "p50": percentile(per_sample_tokens, 0.50),
                "p90": percentile(per_sample_tokens, 0.90),
                "p95": percentile(per_sample_tokens, 0.95),
                "p99": percentile(per_sample_tokens, 0.99),
                "max": per_sample_tokens[-1] if per_sample_tokens else None,
            },
            "length_finish_count": length_finish_count,
            "length_finish_rate": (
                length_finish_count / len(timed_records) if timed_records else None
            ),
            "evaluation_wall_sec": elapsed,
            "output_token_throughput": (
                completion_tokens / elapsed if elapsed > 0 else None
            ),
            "output_token_throughput_definition": (
                "sum(completion_tokens) / pipeline.evaluate() elapsed; "
                "matches the existing MTP report convention"
            ),
            "active_request_union_sec": active_request_union_seconds,
            "active_request_union_output_tok_s": (
                completion_tokens / active_request_union_seconds
                if active_request_union_seconds > 0
                else None
            ),
            "active_request_union_output_definition": (
                "sum(completion_tokens) / union(all non-streaming request wall intervals)"
            ),
        }
        summary_path = args.request_timing_file.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(timing_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    payload = {
        "task": args.task,
        "task_spec": task_spec,
        "elapsed_sec": elapsed,
        "max_samples": args.max_samples,
        "base_url": args.base_url,
        "model": args.model,
        "generation_parameters": generation,
        "http_policy": {
            "parallel_requests": args.parallel,
            "independent_sample_requests": args.independent_samples,
            "timeout_seconds": args.timeout,
            "max_retries_after_initial_attempt": args.max_retries,
            "litellm_num_retries": 0,
        },
        "chat_encoding": {
            "implementation": "sglang.encoding_dsv4",
            "thinking": args.thinking,
            "reasoning_effort": args.reasoning_effort,
        },
        "effective_document_generation_sizes": effective_generation_sizes,
        "metric_names": metric_names,
        "request_timing": timing_summary,
        "results": results,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
