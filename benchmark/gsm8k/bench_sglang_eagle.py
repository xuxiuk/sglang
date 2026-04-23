import argparse
import ast
import json
import os
import re
import time

import numpy as np
from datasets import load_dataset

import sglang as sgl
from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    dump_bench_raw_result,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


@sgl.function
def few_shot_gsm8k(s, question, max_new_tokens):
    s += question
    s += sgl.gen(
        "answer",
        max_tokens=max_new_tokens,
        stop=["Question", "Assistant:", "<|separator|>"],
    )


def main(args):
    # Select backend
    backend = select_sglang_backend(args)
    set_default_backend(backend)

    # Load tokenizer if enable_thinking is set
    tokenizer = None
    if args.enable_thinking:
        from transformers import AutoTokenizer

        assert (
            args.tokenizer_path is not None
        ), "--tokenizer-path is required when --enable-thinking is set"
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

    # Read data
    if args.platinum:
        print("Loading GSM8K Platinum dataset from HuggingFace...")
        dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
        lines = [
            {"question": item["question"], "answer": item["answer"]} for item in dataset
        ]
    else:
        data_path = args.data_path
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        if not os.path.isfile(data_path):
            data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        raw_question = few_shot_examples + get_one_example(lines, i, False)
        if tokenizer is not None:
            messages = [{"role": "user", "content": raw_question}]
            raw_question = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        questions.append(raw_question)
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [
        {"question": q, "max_new_tokens": args.max_new_tokens} for q in questions
    ]

    # Run requests
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=args.temperature,
        top_p=args.top_p,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    preds = [get_answer_value(state["answer"]) for state in states]

    # Compute task metrics
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speculative metrics
    num_output_tokens = sum(
        state.get_meta_info("answer")["completion_tokens"] for state in states
    )
    output_throughput = num_output_tokens / latency if latency > 0 else 0.0

    first_meta = states[0].get_meta_info("answer") if states else {}
    has_verify = "spec_verify_ct" in first_meta
    if has_verify:
        num_verify_ct = sum(
            state.get_meta_info("answer").get("spec_verify_ct", 0) for state in states
        )
        accept_length = num_output_tokens / num_verify_ct if num_verify_ct > 0 else 1.0
    else:
        num_verify_ct = 0
        accept_length = 1.0

    csd_forced_accept_ct = sum(
        state.get_meta_info("answer").get("csd_forced_accept_ct", 0) for state in states
    )
    csd_lookup_hit_ct = sum(
        state.get_meta_info("answer").get("csd_lookup_hit_ct", 0) for state in states
    )
    csd_delta_pair_ct = sum(
        state.get_meta_info("answer").get("csd_delta_pair_ct", 0) for state in states
    )

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Acceptance length: {accept_length:.3f}")
    if args.csd_log_result:
        print(
            "CSD forced_accept: {forced}, lookup_hit: {lookup}, delta_pairs: {delta}".format(
                forced=csd_forced_accept_ct,
                lookup=csd_lookup_hit_ct,
                delta=csd_delta_pair_ct,
            )
        )

    # Dump results
    dump_state_text(args.answer_file or f"tmp_output_{args.backend}.txt", states)
    dump_bench_raw_result(
        path=args.raw_result_file,
        states=states,
        preds=preds,
        labels=labels,
    )

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k-platinum-eagle" if args.platinum else "gsm8k-eagle",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(float(acc), 3),
            "invalid": round(float(invalid), 3),
            "throughput": round(output_throughput, 3),
            "accept_length": round(accept_length, 3),
            "spec_verify_ct": int(num_verify_ct),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
                "num_shots": args.num_shots,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "csd_forced_accept_ct": int(csd_forced_accept_ct),
                "csd_lookup_hit_ct": int(csd_lookup_hit_ct),
                "csd_delta_pair_ct": int(csd_delta_pair_ct),
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode by wrapping prompts with chat template",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (required when --enable-thinking is set)",
    )
    parser.add_argument(
        "--platinum",
        action="store_true",
        help="Use GSM8K Platinum dataset (drop-in replacement with corrected labels)",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=None,
        help="Path to dump decoded answers",
    )
    parser.add_argument(
        "--csd-log-result",
        action="store_true",
        help="Print CSD-related counters from response metadata when available",
    )
    args = add_common_sglang_args_and_parse(parser)
    main(args)
