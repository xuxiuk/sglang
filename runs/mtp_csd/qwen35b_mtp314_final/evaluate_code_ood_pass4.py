#!/usr/bin/env python3
"""Evaluate four APPS/TACO samples per question with the official test runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import inspect
import json
import multiprocessing
import sys
import time
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
from typing import Any


BASE_EVALUATOR = Path(
    "/root/sglang-dflash-csd/benchmark/csd/eval/evaluate_code_ood_accuracy.py"
)
APPS_RAW = Path(
    "/root/.cache/huggingface/hub/datasets--codeparrot--apps/snapshots/"
    "21e74ddf8de1a21436da12e3e653065c5213e9d1/test.jsonl"
)
TACO_RAW = Path(
    "/root/.cache/huggingface/hub/datasets--BAAI--TACO/snapshots/"
    "d593ed0a2becbbc952230bb89be09189bf1056dc/ALL/"
    "test-00000-of-00001.parquet"
)
OFFICIAL_TACO = Path("/tmp/official-taco-eval")


def load_base_evaluator():
    # The pinned APPS/TACO runner still imports the removed IPython
    # ``oinspect.getargspec`` helper.  Keep its expected call surface while
    # running on the newer IPython shipped in the DSpark environment.
    import IPython.core.oinspect as oinspect

    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = getargspec
    if not hasattr(oinspect, "getargspec"):
        oinspect.getargspec = inspect.getargspec

    spec = importlib.util.spec_from_file_location("code_ood_base_evaluator", BASE_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {BASE_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_samples(base: Any, dataset: str, prepared_path: Path, limit: int) -> list[dict]:
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))[:limit]
    if dataset == "apps":
        wanted_ids = {row["source_id"] for row in prepared}
        raw = base.load_apps([APPS_RAW], wanted_ids)
        return base.apps_samples(prepared, raw)
    raw = base.load_taco(TACO_RAW, limit)
    return base.taco_samples(prepared, raw)


def evaluate_answer_file(
    base: Any,
    answer_path: Path,
    samples: list[dict],
    output_path: Path,
    workers: int,
    task_timeout: int,
    expected_n: int,
) -> dict:
    answers = json.loads(answer_path.read_text(encoding="utf-8"))
    expected = len(samples) * expected_n
    if len(answers) != expected:
        raise ValueError(f"{answer_path}: expected {expected} answers, got {len(answers)}")

    jobs = []
    identities = []
    for row in answers:
        question_index = int(row["question_index"])
        sample_index = int(row["sample_index"])
        if not 0 <= question_index < len(samples):
            raise ValueError(f"Invalid question_index={question_index}")
        jobs.append(
            (
                samples[question_index],
                base.extract_python(row.get("output", "")),
                True,
                task_timeout,
            )
        )
        identities.append((question_index, sample_index))

    started = time.time()
    details = []
    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx
    ) as pool:
        futures = [pool.submit(base.evaluate_one, job) for job in jobs]
        for index, future in enumerate(futures):
            question_index, sample_index = identities[index]
            try:
                status, raw_result = future.result(timeout=task_timeout)
            except concurrent.futures.TimeoutError:
                status, raw_result = "global_timeout", [-1]
            except BaseException as exc:
                status = "evaluator_error"
                raw_result = [type(exc).__name__, str(exc)[:300]]
            details.append(
                {
                    "question_index": question_index,
                    "sample_index": sample_index,
                    "status": status,
                    "raw_result": raw_result,
                }
            )
            if (index + 1) % 100 == 0 or index + 1 == len(futures):
                print(f"{answer_path.name}: {index + 1}/{len(futures)}", flush=True)

    by_question: dict[int, list[bool]] = defaultdict(list)
    for row in details:
        by_question[row["question_index"]].append(row["status"] == "passed")
    if len(by_question) != len(samples) or any(
        len(outcomes) != expected_n for outcomes in by_question.values()
    ):
        raise RuntimeError("Incomplete question/sample grouping after evaluation")

    passed_samples = sum(sum(outcomes) for outcomes in by_question.values())
    passed_questions = sum(any(outcomes) for outcomes in by_question.values())
    status_counts = Counter(row["status"] for row in details)
    result = {
        "answer_file": str(answer_path),
        "num_questions": len(samples),
        "samples_per_question": expected_n,
        "num_candidates": len(details),
        "avg_at_n": passed_samples / len(details),
        "pass_at_n": passed_questions / len(samples),
        "metric_names": {"avg": f"pass@1_avg@{expected_n}", "pass": f"pass@{expected_n}"},
        "passed_candidates": passed_samples,
        "passed_questions": passed_questions,
        "status_counts": dict(sorted(status_counts.items())),
        "elapsed_sec": time.time() - started,
        "details": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def method_name(path: Path) -> str:
    name = path.name
    for method in (
        "dynamic_entropy_p20_ignore_ratio",
        "dynamic",
        "plain",
        "eagle",
    ):
        if name.startswith(method + "_"):
            return method
    return name.split("_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("apps", "taco"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--task-timeout", type=int, default=300)
    args = parser.parse_args()

    sys.path.insert(0, str(OFFICIAL_TACO))
    base = load_base_evaluator()
    samples = load_samples(base, args.dataset, args.prepared_data, args.limit)
    answer_paths = sorted((args.run_root / args.dataset / "answers").glob("*_model_outputs.json"))
    if not answer_paths:
        raise SystemExit(f"No answer files under {args.run_root}")

    results = []
    accuracy_dir = args.run_root / args.dataset / "accuracy"
    for answer_path in answer_paths:
        output_path = accuracy_dir / f"{answer_path.stem}.pass{args.n}.json"
        results.append(
            evaluate_answer_file(
                base,
                answer_path,
                samples,
                output_path,
                args.workers,
                args.task_timeout,
                args.n,
            )
        )

    lines = [
        f"# {args.dataset.upper()} n={args.n} official execution accuracy",
        "",
        f"| method | questions | pass@1 avg@{args.n} | pass@{args.n} |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| `{method_name(Path(result['answer_file']))}` | {result['num_questions']} | "
            f"{100 * result['avg_at_n']:.2f}% | {100 * result['pass_at_n']:.2f}% |"
        )
    (accuracy_dir / f"summary_pass{args.n}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
