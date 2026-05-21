import argparse
import fcntl
import gzip
import io
import json
import os
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

import requests

import sglang as sgl
from sglang.lang.api import set_default_backend
from sglang.test.test_utils import add_common_sglang_args_and_parse, select_sglang_backend


REDPAJAMA_DOMAINS = [
    "arxiv",
    "c4",
    "common_crawl",
    "github",
    "stackexchange",
    "wikipedia",
]


REDPAJAMA_URL_LISTS = {
    "arxiv": "urls/arxiv.txt",
    "c4": "urls/c4.txt",
    "common_crawl": "urls/common_crawl.txt",
    "github": "urls/github.txt",
    "stackexchange": "urls/stackexchange.txt",
    "wikipedia": "urls/wikipedia.txt",
}


REDPAJAMA_REPO_RAW_BASE = "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/raw/main"
REDPAJAMA_MIRROR_RAW_BASE = "https://hf-mirror.com/datasets/togethercomputer/RedPajama-Data-1T/raw/main"
URL_HEADERS = {"User-Agent": "Mozilla/5.0"}


TEXT_FIELDS = ("text", "content", "raw_content", "document")


def _safe_filename_part(value):
    if value is None:
        return "none"
    value = str(value).strip()
    if not value:
        return "none"
    value = value.rstrip("/").split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value.strip("-_") or "none"


def _draft_model_name(args):
    return args.draft_model_name or "mtp"


def _speculative_config(args):
    return {
        "algorithm": args.speculative_algorithm,
        "num_steps": args.speculative_num_steps,
        "eagle_topk": args.speculative_eagle_topk,
        "num_draft_tokens": args.speculative_num_draft_tokens,
    }


def _csd_config(args):
    return {
        "enabled": args.csd_enabled,
        "dynamic_update": args.csd_dynamic_update,
        "force_accept_disabled": args.csd_force_accept_disabled,
        "freq_threshold": args.csd_freq_threshold,
        "prob_ratio": args.csd_prob_ratio,
        "save_table_path": args.csd_save_table_path,
    }


def _default_csd_table_path(args):
    parts = [
        "csd",
        "redpajama",
        f"n{args.samples_per_domain}_x{len(args.domains)}",
        _safe_filename_part(args.model_name),
        _safe_filename_part(_draft_model_name(args)),
        _safe_filename_part(args.speculative_algorithm),
        f"temp{args.temperature:g}",
    ]
    if args.run_tag:
        parts.append(args.run_tag)
    return str(Path(args.csd_save_dir) / ("_".join(_safe_filename_part(part) for part in parts) + ".json"))


def _redpajama_cache_dir():
    return Path(os.environ.get("REDPAJAMA_URL_CACHE_DIR", "/home/zhouxuwen/sglang-debug/benchmark/csd/runs/redpajama_url_cache"))


def _read_cached_url_list(path):
    cache_path = _redpajama_cache_dir() / path
    if cache_path.exists():
        return [line.strip() for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None


def _write_cached_url_list(path, urls):
    cache_path = _redpajama_cache_dir() / path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def _open_url_text(url):
    response = urlopen(Request(url, headers=URL_HEADERS), timeout=60)
    if url.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading RedPajama common_crawl requires the zstandard package."
            ) from exc
        return zstd.open(response, "rt", encoding="utf-8")
    if url.endswith(".gz"):
        return gzip.open(response, "rt", encoding="utf-8")
    return io.TextIOWrapper(response, encoding="utf-8")


def _download_url_list(path):
    bases = []
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint and "hf-mirror.com" in endpoint:
        bases.append(REDPAJAMA_MIRROR_RAW_BASE)
    bases.append(REDPAJAMA_REPO_RAW_BASE)
    if REDPAJAMA_MIRROR_RAW_BASE not in bases:
        bases.append(REDPAJAMA_MIRROR_RAW_BASE)

    last_error = None
    for _ in range(3):
        for base in bases:
            try:
                with urlopen(Request(f"{base}/{path}", headers=URL_HEADERS), timeout=30) as response:
                    urls = [
                        line.strip()
                        for line in response.read().decode("utf-8").splitlines()
                        if line.strip()
                    ]
                    _write_cached_url_list(path, urls)
                    return urls
            except Exception as exc:
                last_error = exc
            time.sleep(2)
    raise RuntimeError(f"Failed to load RedPajama URL list {path}: {last_error}")


def _read_url_list(path):
    cached_urls = _read_cached_url_list(path)
    if cached_urls is not None:
        return cached_urls

    lock_path = _redpajama_cache_dir() / "url_lists.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        cached_urls = _read_cached_url_list(path)
        if cached_urls is not None:
            return cached_urls
        return _download_url_list(path)


def _iter_domain_examples(args, domain):
    if domain not in REDPAJAMA_URL_LISTS:
        raise ValueError(
            f"Unsupported RedPajama domain {domain}. Supported domains: {sorted(REDPAJAMA_URL_LISTS)}"
        )
    urls = _read_url_list(REDPAJAMA_URL_LISTS[domain])
    for url in urls:
        rows = _open_url_text(url)
        try:
            for row in rows:
                if not row:
                    continue
                data = json.loads(row)
                data.setdefault("red_pajama_subset", domain)
                yield data
        finally:
            close = getattr(rows, "close", None)
            if close is not None:
                close()


def _extract_text(example):
    for field in TEXT_FIELDS:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value
    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value
    return None


def _truncate_prompt(text, max_chars):
    text = " ".join(text.split())
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def load_prompts(args):
    prompts = []
    per_domain_counts = {}
    for domain in args.domains:
        dataset = _iter_domain_examples(args, domain)
        count = 0
        for example in dataset:
            text = _extract_text(example)
            if text is None:
                continue
            prompt = _truncate_prompt(text, args.prompt_chars)
            if len(prompt) < args.min_prompt_chars:
                continue
            prompts.append({"domain": domain, "prompt": prompt})
            count += 1
            if count >= args.samples_per_domain:
                break
        if count < args.samples_per_domain:
            raise RuntimeError(
                f"Domain {domain} only produced {count} usable prompts; expected {args.samples_per_domain}"
            )
        per_domain_counts[domain] = count
    return prompts, per_domain_counts


@sgl.function
def continue_text(s, prompt, max_new_tokens):
    s += prompt
    s += sgl.gen("completion", max_tokens=max_new_tokens)


def _save_csd_table(args, metadata):
    if not args.csd_save_table_path:
        if not args.csd_auto_save_table:
            return
        args.csd_save_table_path = _default_csd_table_path(args)

    Path(args.csd_save_table_path).parent.mkdir(parents=True, exist_ok=True)
    metadata["csd"]["save_table_path"] = args.csd_save_table_path

    response = requests.post(
        f"http://{args.host}:{args.port}/save_csd_table",
        json={"path": args.csd_save_table_path, "metadata": metadata},
        timeout=args.csd_save_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", False):
        raise RuntimeError(payload.get("message", "failed to save CSD table"))
    print(f"Saved CSD table to {args.csd_save_table_path}")


def main(args):
    backend = select_sglang_backend(args)
    set_default_backend(backend)

    prompts, per_domain_counts = load_prompts(args)
    arguments = [
        {"prompt": item["prompt"], "max_new_tokens": args.max_new_tokens}
        for item in prompts
    ]

    tic = time.perf_counter()
    states = continue_text.run_batch(
        arguments,
        temperature=args.temperature,
        top_p=args.top_p,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    metas = [state.get_meta_info("completion") for state in states]
    num_output_tokens = sum(meta.get("completion_tokens", 0) for meta in metas)
    output_throughput = num_output_tokens / latency if latency > 0 else 0.0
    num_verify_ct = sum(meta.get("spec_verify_ct", 0) for meta in metas)
    accept_length = num_output_tokens / num_verify_ct if num_verify_ct > 0 else 1.0
    spec_accept_token_num = max(num_output_tokens - num_verify_ct, 0)
    spec_draft_token_num = num_verify_ct * (args.speculative_num_steps or 0)
    spec_success_rate = (
        (accept_length - 1) / args.speculative_num_steps
        if args.speculative_num_steps
        else 0.0
    )
    spec_output_token_saved_ratio = (
        (num_output_tokens - num_verify_ct) / num_output_tokens
        if num_output_tokens > 0
        else 0.0
    )
    csd_forced_accept_ct = max(
        (meta.get("csd_forced_accept_ct", 0) for meta in metas), default=0
    )
    csd_lookup_hit_ct = max(
        (meta.get("csd_lookup_hit_ct", 0) for meta in metas), default=0
    )
    csd_delta_pair_ct = max(
        (meta.get("csd_delta_pair_ct", 0) for meta in metas), default=0
    )

    metadata = {
        "task": "redpajama-csd-calibration",
        "dataset": args.dataset_name,
        "domains": args.domains,
        "samples_per_domain": args.samples_per_domain,
        "per_domain_counts": per_domain_counts,
        "num_prompts": len(prompts),
        "prompt_chars": args.prompt_chars,
        "min_prompt_chars": args.min_prompt_chars,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "parallel": args.parallel,
        "backend": args.backend,
        "host": args.host,
        "port": args.port,
        "model": args.model_name,
        "draft_model": _draft_model_name(args),
        "speculative": _speculative_config(args),
        "run_tag": args.run_tag,
        "csd": _csd_config(args),
        "latency": round(latency, 3),
        "throughput": round(output_throughput, 3),
        "accept_length": round(accept_length, 3),
        "spec_success_rate": round(spec_success_rate, 6),
        "spec_output_token_saved_ratio": round(spec_output_token_saved_ratio, 6),
        "spec_verify_ct": int(num_verify_ct),
        "spec_accept_token_num": int(spec_accept_token_num),
        "spec_draft_token_num": int(spec_draft_token_num),
        "csd_forced_accept_ct": int(csd_forced_accept_ct),
        "csd_lookup_hit_ct": int(csd_lookup_hit_ct),
        "csd_delta_pair_ct": int(csd_delta_pair_ct),
    }

    print(f"Prompts: {len(prompts)}")
    print(f"Per-domain counts: {per_domain_counts}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Acceptance length: {accept_length:.3f}")
    print(f"Speculative success rate: {spec_success_rate:.3f}")
    print(f"Speculative output token saved ratio: {spec_output_token_saved_ratio:.3f}")
    if args.csd_log_result:
        print(
            "CSD forced_accept: {forced}, lookup_hit: {lookup}, delta_pairs: {delta}".format(
                forced=csd_forced_accept_ct,
                lookup=csd_lookup_hit_ct,
                delta=csd_delta_pair_ct,
            )
        )

    if args.answer_file:
        Path(args.answer_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.answer_file, "w") as fout:
            fout.write("# " + json.dumps({"metadata": metadata}) + "\n")
            for item, state in zip(prompts, states):
                fout.write(
                    json.dumps(
                        {
                            "domain": item["domain"],
                            "prompt": item["prompt"],
                            "completion": state["completion"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    _save_csd_table(args, metadata)

    if args.result_file:
        Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_file, "a") as fout:
            value = {
                "task": metadata["task"],
                "backend": args.backend,
                "num_gpus": 1,
                "latency": metadata["latency"],
                "throughput": metadata["throughput"],
                "accept_length": metadata["accept_length"],
                "spec_success_rate": metadata["spec_success_rate"],
                "spec_output_token_saved_ratio": metadata[
                    "spec_output_token_saved_ratio"
                ],
                "spec_verify_ct": metadata["spec_verify_ct"],
                "spec_accept_token_num": metadata["spec_accept_token_num"],
                "spec_draft_token_num": metadata["spec_draft_token_num"],
                "num_requests": len(prompts),
                "other": metadata,
            }
            fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="togethercomputer/RedPajama-Data-1T",
        help="Stored in metadata; RedPajama is read directly from URL lists.",
    )
    parser.add_argument("--domains", nargs="+", default=REDPAJAMA_DOMAINS)
    parser.add_argument("--samples-per-domain", type=int, default=1500)
    parser.add_argument("--prompt-chars", type=int, default=4096)
    parser.add_argument("--min-prompt-chars", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--draft-model-name", type=str, default=None)
    parser.add_argument("--speculative-algorithm", type=str, default="EAGLE")
    parser.add_argument("--speculative-num-steps", type=int, default=None)
    parser.add_argument("--speculative-eagle-topk", type=int, default=None)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=None)
    parser.add_argument("--csd-enabled", action="store_true")
    parser.add_argument("--csd-dynamic-update", action="store_true")
    parser.add_argument("--csd-force-accept-disabled", action="store_true")
    parser.add_argument("--csd-freq-threshold", type=int, default=None)
    parser.add_argument("--csd-prob-ratio", type=float, default=None)
    parser.add_argument("--csd-log-result", action="store_true")
    parser.add_argument("--csd-save-table-path", type=str, default=None)
    parser.add_argument("--csd-auto-save-table", action="store_true")
    parser.add_argument("--csd-save-dir", type=str, default="benchmark/csd/runs/redpajama")
    parser.add_argument("--csd-save-timeout", type=float, default=120.0)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
