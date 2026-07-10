"""
Adapted from https://github.com/chromecast56/sglang/blob/6f145d2eadb93a116134f703358ce76f15381045/benchmark/mtbench/bench_sglang.py

Benchmark SGLang EAGLE/EAGLE3 Speculative Decoding

Usage:
python3 benchmark/mtbench/bench_sglang_eagle.py --num-questions 80 --parallel 1
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import uuid

import requests
from sglang.test.test_utils import add_common_sglang_args_and_parse
from sglang.utils import download_and_cache_file
from transformers import AutoTokenizer


def load_questions(filename):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions


def write_answers(filename, model_id, questions, answers):
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(answers)):
            ans_json = {
                "question_id": questions[i]["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": {
                    "index": 0,
                    "turns": [answers[i][0], answers[i][1]],
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def build_prompt(tokenizer, messages, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def call_generate(url: str, prompt: str, sampling: dict) -> dict:
    response = requests.post(
        url,
        json={"text": prompt, "sampling_params": sampling},
        timeout=None,
    )
    response.raise_for_status()
    return response.json()


def meta_token_count(meta: dict) -> tuple[int, int | None]:
    completion_tokens = int(meta.get("completion_tokens", 0))
    verify_ct = meta.get("spec_verify_ct")
    if verify_ct is not None:
        verify_ct = int(verify_ct)
    return completion_tokens, verify_ct


def main(args):
    # Download question file if not exist
    question_file = args.question_file
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    if not os.path.isfile(question_file):
        question_file = download_and_cache_file(url)

    questions = load_questions(question_file)[: args.num_questions]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

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

    base_messages = []
    for q in questions:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": q["turns"][0]})
        base_messages.append(messages)

    url = f"http://{args.host}:{args.port}/generate"
    tic = time.perf_counter()
    first_prompts = [
        build_prompt(tokenizer, messages, args.enable_thinking)
        for messages in base_messages
    ]
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        first_rets = list(
            executor.map(lambda item: call_generate(url, item, sampling), first_prompts)
        )

    second_messages = []
    for q, messages, ret in zip(questions, base_messages, first_rets):
        history = list(messages)
        history.append({"role": "assistant", "content": ret["text"]})
        history.append({"role": "user", "content": q["turns"][1]})
        second_messages.append(history)

    second_prompts = [
        build_prompt(tokenizer, messages, args.enable_thinking)
        for messages in second_messages
    ]
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        second_rets = list(
            executor.map(lambda item: call_generate(url, item, sampling), second_prompts)
        )
    answers = [
        [first["text"], second["text"]]
        for first, second in zip(first_rets, second_rets)
    ]

    latency = time.perf_counter() - tic
    completion_tokens = []
    verify_cts = []
    for ret in first_rets + second_rets:
        completion, verify_ct = meta_token_count(ret["meta_info"])
        completion_tokens.append(completion)
        if verify_ct is not None:
            verify_cts.append(verify_ct)

    num_output_tokens = sum(completion_tokens)
    output_throughput = num_output_tokens / latency

    accept_length = (
        num_output_tokens / sum(verify_cts)
        if verify_cts and sum(verify_cts) > 0
        else 1.0
    )

    print(
        f"#questions: {len(questions)}, Throughput: {output_throughput:.2f} token/s, Acceptance length: {accept_length:.2f}"
    )

    # Write results
    model_id = args.model_id or args.tokenizer_path
    answer_file = args.answer_file or f"tmp_output_{args.backend}.txt"
    write_answers(answer_file, model_id, questions, answers)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "mtbench",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "throughput": round(output_throughput, 3),
            "accept_length": round(accept_length, 3),
            "num_requests": args.num_questions,
            "total_completion_tokens": num_output_tokens,
            "avg_completion_tokens": round(num_output_tokens / (2 * len(questions)), 3)
            if questions
            else 0.0,
            "max_completion_tokens": max(completion_tokens) if completion_tokens else 0,
            "max_new_token_hits": sum(
                1 for count in completion_tokens if count >= args.max_new_tokens
            ),
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
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
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=80)
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
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
