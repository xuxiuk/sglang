"""
Custom lm-eval backend for SGLang that uses the native /generate endpoint.

Collects speculative decoding and CSD metrics from each response's meta_info,
alongside standard accuracy evaluation. Use run_lm_eval_sglang_native.py as the
entry point so the model can be instantiated directly without registering into
site-packages/lm_eval/models.
"""

import json
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.models.utils import handle_stop_sequences


eval_logger = logging.getLogger("lm_eval")


@register_model("sglang-native")
class SGLangNativeLM(LocalCompletionsAPI):
    """SGLang native /generate backend that collects speculative/CSD metrics."""

    def __init__(
        self,
        base_url=None,
        tokenizer_backend="huggingface",
        collect_spec_metrics=True,
        metrics_output_path=None,
        speculative_num_steps=None,
        **kwargs,
    ):
        kwargs.setdefault("tokenized_requests", False)
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            **kwargs,
        )
        self.collect_spec_metrics = collect_spec_metrics
        self.speculative_num_steps = speculative_num_steps or 0
        self.metrics_output_path = (
            metrics_output_path or "lm_eval_sglang_native_metrics.json"
        )

        # Thread-safe speculative metrics accumulator
        self._spec_metrics: List[Dict] = []
        self._metrics_lock = threading.Lock()

        # Point to /generate endpoint
        if base_url and "/generate" not in base_url:
            self.base_url = base_url.rstrip("/") + "/generate"

    def _chat_messages_to_text(self, messages):
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )

        parts = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            parts.append(str(content))
        return "\n".join(parts)

    def _normalize_messages(self, messages):
        if isinstance(messages, list) and messages:
            if isinstance(messages[0], dict):
                return self._chat_messages_to_text(messages)
            if (
                isinstance(messages[0], list)
                and messages[0]
                and isinstance(messages[0][0], dict)
            ):
                return [self._chat_messages_to_text(message) for message in messages]
        return messages

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        messages = self._normalize_messages(messages)
        is_string = (
            True
            if (isinstance(messages, str) or isinstance(messages[0], str))
            else False
        )
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            request = {
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    **gen_kwargs,
                },
            }
            request.update({"text": messages}) if is_string else request.update(
                {"input_ids": messages}
            )
            return request
        else:
            assert not is_string, "Logprobs are only supported for tokenized inputs"
            return {
                "input_ids": messages,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "logprob_start_len": 0,
                "top_logprobs_num": 1,
                "return_logprob": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for choice, ctxlen in zip(outputs, ctxlens):
            choice = choice["meta_info"]
            assert ctxlen > 0, "Context length must be greater than 0"
            logprobs = sum(x[0] for x in choice["input_token_logprobs"][ctxlen:])
            is_greedy = all(
                x[1] != y[0][1]
                for x, y in zip(
                    choice["input_token_logprobs"][ctxlen:],
                    choice["input_top_logprobs"][ctxlen:],
                )
            )
            res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(
        outputs: Union[Dict, List[Dict]], **kwargs
    ) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            res.append(out["text"])
        return res

    @property
    def api_key(self):
        return ""

    def _extract_metrics_from_outputs(self, outputs):
        """Extract speculative/CSD metrics from raw /generate responses."""
        if not self.collect_spec_metrics:
            return
        if not isinstance(outputs, list):
            outputs = [outputs]
        entries = []
        for out in outputs:
            meta = out.get("meta_info", {})
            if not meta:
                continue
            spec_accept_length = meta.get("spec_accept_length", meta.get("accept_length", 0))
            spec_verify_ct = meta.get("spec_verify_ct", 0)
            spec_num_steps = meta.get("speculative_num_steps") or self.speculative_num_steps
            entry = {
                "spec_accept_length": spec_accept_length,
                "spec_accept_rate": meta.get("spec_accept_rate", 0),
                "spec_accept_token_num": meta.get("spec_accept_token_num", 0),
                "spec_draft_token_num": meta.get("spec_draft_token_num", 0),
                "spec_verify_ct": spec_verify_ct,
                "speculative_num_steps": spec_num_steps,
                "speculative_num_draft_tokens": meta.get(
                    "speculative_num_draft_tokens", 0
                ),
                "csd_lookup_hit_ct": meta.get("csd_lookup_hit_ct", 0),
                "csd_forced_accept_ct": meta.get("csd_forced_accept_ct", 0),
                "csd_delta_pair_ct": meta.get("csd_delta_pair_ct", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
                "prompt_tokens": meta.get("prompt_tokens", 0),
            }
            entries.append(entry)
        with self._metrics_lock:
            self._spec_metrics.extend(entries)

    def model_call(self, *args, **kwargs):
        """Intercept sync generation responses to collect speculative metrics."""
        result = super().model_call(*args, **kwargs)
        if result is not None and kwargs.get("generate", True):
            self._extract_metrics_from_outputs(result)
        return result

    async def amodel_call(
        self,
        session,
        sem,
        messages,
        *,
        generate=True,
        cache_keys=None,
        ctxlens=None,
        gen_kwargs=None,
        **kwargs,
    ):
        """Async API call that collects metrics before parsing outputs."""
        import copy

        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        acquired = await sem.acquire()
        try:
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    eval_logger.warning(
                        f"API request failed! Status code: {response.status}, "
                        f"Response text: {error_text}. Retrying..."
                    )
                response.raise_for_status()
                outputs = await response.json()

            if generate:
                self._extract_metrics_from_outputs(outputs)

            answers = (
                self.parse_generations(outputs=outputs)
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )
            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        except BaseException as e:
            eval_logger.error(f"Exception:{repr(e)}, retrying.")
            raise e
        finally:
            if acquired:
                sem.release()

    def save_spec_metrics(self):
        """Print and save collected speculative metrics."""
        with self._metrics_lock:
            metrics = list(self._spec_metrics)

        if not metrics:
            eval_logger.warning("No speculative metrics collected.")
            return None

        total = len(metrics)
        avg_accept_length = sum(m["spec_accept_length"] for m in metrics) / total
        avg_accept_rate = sum(m["spec_accept_rate"] for m in metrics) / total
        total_csd_hits = sum(m["csd_lookup_hit_ct"] for m in metrics)
        total_csd_forced = sum(m["csd_forced_accept_ct"] for m in metrics)
        total_csd_delta = sum(m["csd_delta_pair_ct"] for m in metrics)
        total_prompt_tokens = sum(m["prompt_tokens"] for m in metrics)
        total_completion_tokens = sum(m["completion_tokens"] for m in metrics)
        total_spec_verify_ct = sum(m["spec_verify_ct"] for m in metrics)
        total_spec_accept_tokens = sum(m["spec_accept_token_num"] for m in metrics)
        total_spec_draft_tokens = sum(m["spec_draft_token_num"] for m in metrics)
        spec_num_steps = max((m["speculative_num_steps"] for m in metrics), default=0)
        aggregate_accept_rate = (
            total_spec_accept_tokens / total_spec_draft_tokens
            if total_spec_draft_tokens
            else 0
        )
        aggregate_accept_length = (
            total_completion_tokens / total_spec_verify_ct
            if total_spec_verify_ct
            else 0
        )
        spec_success_rate = (
            (aggregate_accept_length - 1) / spec_num_steps
            if spec_num_steps and total_spec_verify_ct
            else 0
        )
        spec_output_token_saved_ratio = (
            total_spec_accept_tokens / (total_spec_accept_tokens + total_spec_verify_ct)
            if total_spec_accept_tokens + total_spec_verify_ct
            else 0
        )

        summary = {
            "total_requests": total,
            "avg_spec_accept_length": round(avg_accept_length, 4),
            "aggregate_spec_accept_length": round(aggregate_accept_length, 4),
            "avg_spec_accept_rate": round(avg_accept_rate, 4),
            "aggregate_spec_accept_rate": round(aggregate_accept_rate, 4),
            "spec_success_rate": round(spec_success_rate, 4),
            "avg_spec_success_rate": round(spec_success_rate, 4),
            "speculative_num_steps": spec_num_steps,
            "spec_output_token_saved_ratio": round(spec_output_token_saved_ratio, 4),
            "total_spec_verify_ct": total_spec_verify_ct,
            "total_spec_accept_token_num": total_spec_accept_tokens,
            "total_spec_draft_token_num": total_spec_draft_tokens,
            "total_csd_lookup_hits": total_csd_hits,
            "total_csd_forced_accepts": total_csd_forced,
            "total_csd_delta_pairs": total_csd_delta,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "per_request_metrics": metrics,
        }

        print("\n" + "=" * 60)
        print("SGLang Speculative Decoding Metrics")
        print("=" * 60)
        print(f"Total requests:           {total}")
        print(f"Avg spec accept length:   {avg_accept_length:.4f}")
        print(f"Avg spec accept rate:     {avg_accept_rate:.4f}")
        print(f"Aggregate accept rate:    {aggregate_accept_rate:.4f}")
        print(f"Output token saved ratio: {spec_output_token_saved_ratio:.4f}")
        print(f"Total spec verify ct:     {total_spec_verify_ct}")
        print(f"Total spec accept tokens: {total_spec_accept_tokens}")
        print(f"Total spec draft tokens:  {total_spec_draft_tokens}")
        print(f"Total CSD lookup hits:    {total_csd_hits}")
        print(f"Total CSD forced accepts: {total_csd_forced}")
        print(f"Total CSD delta pairs:    {total_csd_delta}")
        print(f"Total prompt tokens:      {total_prompt_tokens}")
        print(f"Total completion tokens:  {total_completion_tokens}")
        print("=" * 60 + "\n")

        with open(self.metrics_output_path, "w") as f:
            json.dump(summary, f, indent=2)
        eval_logger.info(
            f"Speculative metrics saved to {self.metrics_output_path}"
        )
        return summary
