#!/usr/bin/env python3
"""Generate LongBench answers with SGLang and record throughput/spec metrics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from pathlib import Path
from typing import Any
import urllib.request
import zipfile
from collections import defaultdict

import requests
from sglang.test.test_utils import add_common_sglang_args_and_parse
from transformers import AutoTokenizer


PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can.\n\nStory: {context}\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question based on the article. If the answer is yes or no, answer yes or no. If the answer is not in the article, answer unanswerable.\n\nArticle: {context}\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer the question briefly.\n\nText: {context}\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_zh": "请阅读以下长文本，并简洁回答问题。\n\n文本：{context}\n\n问题：{input}\n\n答案：",
    "hotpotqa": "Answer the question based on the given passages.\n\nPassages: {context}\n\nQuestion: {input}\n\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages.\n\nPassages: {context}\n\nQuestion: {input}\n\nAnswer:",
    "musique": "Answer the question based on the given passages.\n\nPassages: {context}\n\nQuestion: {input}\n\nAnswer:",
    "gov_report": "You are given a government report. Write a concise summary of the report.\n\nReport: {context}\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query. Summarize the transcript according to the query.\n\nTranscript: {context}\n\nQuery: {input}\n\nSummary:",
    "multi_news": "You are given several news articles. Write a concise summary of the articles.\n\nArticles: {context}\n\nSummary:",
    "trec": "Classify the question into one of the provided classes.\n\nQuestion: {input}\n\nClass:",
    "triviaqa": "Answer the question based on the given evidence.\n\nEvidence: {context}\n\nQuestion: {input}\n\nAnswer:",
    "dureader": "请根据给定材料回答问题。\n\n材料：{context}\n\n问题：{input}\n\n答案：",
    "samsum": "Summarize the following dialogue.\n\nDialogue: {context}\n\nSummary:",
    "vcsum": "请总结以下内容。\n\n内容：{context}\n\n摘要：",
    "lsht": "请根据给定新闻内容判断其类别。\n\n新闻：{context}\n\n类别：",
    "passage_count": "There are several paragraphs below. Count how many unique paragraphs there are.\n\nParagraphs: {context}\n\nAnswer:",
    "passage_retrieval_en": "You are given several passages and a question. Find the passage that answers the question.\n\nPassages: {context}\n\nQuestion: {input}\n\nAnswer:",
    "passage_retrieval_zh": "给定若干段落和一个问题，请找出能够回答问题的段落。\n\n段落：{context}\n\n问题：{input}\n\n答案：",
    "lcc": "Please complete the code.\n\n{context}\n\n{input}",
    "repobench-p": "Please complete the code.\n\n{context}\n\n{input}",
}

LONGBENCH_V1_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

LONGBENCH_E_TASKS = [
    "qasper_e",
    "multifieldqa_en_e",
    "hotpotqa_e",
    "2wikimqa_e",
    "gov_report_e",
    "multi_news_e",
    "trec_e",
    "triviaqa_e",
    "samsum_e",
    "passage_count_e",
    "passage_retrieval_en_e",
    "lcc_e",
    "repobench-p_e",
]


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def ensure_data_zip(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    url = f"{endpoint}/datasets/THUDM/LongBench/resolve/main/data.zip"
    urllib.request.urlretrieve(url, path)
    return path


def load_rows(task: str, limit: int | None, offset: int, data_zip: Path) -> list[dict[str, Any]]:
    data_zip = ensure_data_zip(data_zip)
    candidates = [
        f"data/{task}.jsonl",
        f"{task}.jsonl",
        f"LongBench/data/{task}.jsonl",
    ]
    with zipfile.ZipFile(data_zip) as zf:
        names = set(zf.namelist())
        filename = next((name for name in candidates if name in names), None)
        if filename is None:
            matches = [name for name in names if name.endswith(f"/{task}.jsonl")]
            if matches:
                filename = matches[0]
        if filename is None:
            raise FileNotFoundError(
                f"Could not find {task}.jsonl in {data_zip}; examples: {sorted(names)[:10]}"
            )
        with zf.open(filename) as fin:
            rows = [
                json.loads(line.decode("utf-8"))
                for line in fin
                if line.strip()
            ]
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def canonical_task(task: str) -> str:
    return task[:-2] if task.endswith("_e") else task


def expand_tasks(tasks_arg: str, data_zip: Path) -> list[str]:
    raw_tasks = [task.strip() for task in tasks_arg.split(",") if task.strip()]
    expanded: list[str] = []
    for task in raw_tasks:
        if task == "all":
            expanded.extend(LONGBENCH_V1_TASKS)
            expanded.extend(LONGBENCH_E_TASKS)
        elif task in {"all_v1", "v1"}:
            expanded.extend(LONGBENCH_V1_TASKS)
        elif task in {"all_e", "longbench_e"}:
            expanded.extend(LONGBENCH_E_TASKS)
        elif task == "all_zip":
            data_zip = ensure_data_zip(data_zip)
            with zipfile.ZipFile(data_zip) as zf:
                expanded.extend(
                    sorted(
                        Path(name).stem
                        for name in zf.namelist()
                        if name.endswith(".jsonl")
                    )
                )
        else:
            expanded.append(task)
    seen = set()
    deduped = []
    for task in expanded:
        if task not in seen:
            deduped.append(task)
            seen.add(task)
    return deduped


def render_prompt(task: str, row: dict[str, Any]) -> str:
    template = PROMPTS.get(canonical_task(task))
    if template is None:
        raise ValueError(f"Unsupported LongBench task: {task}")
    return template.format(
        context=row.get("context", ""),
        input=row.get("input", ""),
    )


def build_prompt(
    tokenizer: Any, prompt: str, system_prompt: str, enable_thinking: bool
) -> str:
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
    parser.add_argument("--longbench-tasks", required=True)
    parser.add_argument(
        "--longbench-data-zip",
        default="/root/sglang/benchmark/csd/runs/longbench_cache/data.zip",
    )
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--num-examples-per-task", type=int, default=None)
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
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = add_common_sglang_args_and_parse(parser)

    data_zip = Path(args.longbench_data_zip)
    tasks = expand_tasks(args.longbench_tasks, data_zip)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    examples = []
    prompts = []
    prompt_token_counts = []
    for task in tasks:
        rows = load_rows(task, args.num_examples_per_task, args.offset, data_zip)
        for idx, row in enumerate(rows):
            prompt = render_prompt(task, row)
            chat_prompt = build_prompt(
                tokenizer, prompt, args.system_prompt, args.enable_thinking
            )
            examples.append((task, idx + args.offset, row))
            prompts.append(chat_prompt)
            prompt_token_counts.append(len(tokenizer.encode(chat_prompt)))

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
    per_task_stats: dict[str, dict[str, Any]] = defaultdict(
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
        for (task, idx, row), ret, prompt_tokens in zip(
            examples, rets, prompt_token_counts
        ):
            completion, verify_ct = meta_token_count(ret["meta_info"])
            completion_tokens.append(completion)
            if verify_ct is not None:
                verify_cts.append(verify_ct)
            stats = per_task_stats[task]
            stats["num_requests"] += 1
            stats["completion_tokens"].append(completion)
            stats["prompt_tokens"].append(prompt_tokens)
            if verify_ct is not None:
                stats["verify_cts"].append(verify_ct)
            fout.write(
                json.dumps(
                    {
                        "task": task,
                        "index": idx,
                        "prediction": ret["text"],
                        "answers": row.get("answers", []),
                        "all_classes": row.get("all_classes", []),
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
        "task": "longbench:" + ",".join(tasks),
        "model_id": model_id,
        "backend": args.backend,
        "num_gpus": None,
        "latency": round(latency, 3),
        "throughput": round(throughput, 3),
        "accept_length": round(accept_length, 3),
        "num_requests": len(examples),
        "total_completion_tokens": total_completion_tokens,
        "avg_completion_tokens": round(total_completion_tokens / len(examples), 3)
        if examples
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
            "longbench_tasks": tasks,
            "num_examples_per_task": args.num_examples_per_task,
            "parallel": args.parallel,
            "answer_file": str(answer_path),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "system_prompt": args.system_prompt,
            "enable_thinking": args.enable_thinking,
            "tokenizer_path": args.tokenizer_path,
        },
    }

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        for task in tasks:
            stats = per_task_stats[task]
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
                "task": "longbench:" + task,
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
                    "parent_task": result["task"],
                    "longbench_task": task,
                    "num_examples_per_task": args.num_examples_per_task,
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
