# MIT License

import json
import logging
import threading
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class SGLangMetricsAccumulator:
    def __init__(self, collect: bool = True, speculative_num_steps: int | None = None):
        self.collect = collect
        self.speculative_num_steps = speculative_num_steps or 0
        self._metrics: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_outputs(self, outputs: dict[str, Any] | list[dict[str, Any]] | None) -> None:
        if not self.collect or outputs is None:
            return
        if not isinstance(outputs, list):
            outputs = [outputs]

        entries = []
        for out in outputs:
            if not isinstance(out, dict):
                continue
            meta = out.get("meta_info", {})
            if not meta:
                continue
            spec_accept_length = meta.get("spec_accept_length", meta.get("accept_length", 0))
            spec_num_steps = meta.get("speculative_num_steps") or self.speculative_num_steps
            entries.append(
                {
                    "spec_accept_length": spec_accept_length,
                    "spec_accept_rate": meta.get("spec_accept_rate", 0),
                    "spec_accept_token_num": meta.get("spec_accept_token_num", 0),
                    "spec_draft_token_num": meta.get("spec_draft_token_num", 0),
                    "spec_verify_ct": meta.get("spec_verify_ct", 0),
                    "speculative_num_steps": spec_num_steps,
                    "speculative_num_draft_tokens": meta.get("speculative_num_draft_tokens", 0),
                    "csd_lookup_hit_ct": meta.get("csd_lookup_hit_ct", 0),
                    "csd_forced_accept_ct": meta.get("csd_forced_accept_ct", 0),
                    "csd_delta_pair_ct": meta.get("csd_delta_pair_ct", 0),
                    "completion_tokens": meta.get("completion_tokens", 0),
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                }
            )

        if entries:
            with self._lock:
                self._metrics.extend(entries)

    def summary(self) -> dict[str, Any] | None:
        with self._lock:
            metrics = list(self._metrics)

        if not metrics:
            logger.warning("No speculative metrics collected.")
            return None

        total = len(metrics)
        avg_accept_length = sum(m["spec_accept_length"] for m in metrics) / total
        total_csd_hits = sum(m["csd_lookup_hit_ct"] for m in metrics)
        total_csd_forced = sum(m["csd_forced_accept_ct"] for m in metrics)
        total_csd_delta = sum(m["csd_delta_pair_ct"] for m in metrics)
        total_prompt_tokens = sum(m["prompt_tokens"] for m in metrics)
        total_completion_tokens = sum(m["completion_tokens"] for m in metrics)
        total_spec_verify_ct = sum(m["spec_verify_ct"] for m in metrics)
        total_spec_accept_tokens = sum(m["spec_accept_token_num"] for m in metrics)
        total_spec_draft_tokens = sum(m["spec_draft_token_num"] for m in metrics)
        spec_num_steps = max((m["speculative_num_steps"] for m in metrics), default=0)
        avg_accept_rate = (
            sum(
                (m["spec_accept_length"] - 1) / m["speculative_num_steps"]
                for m in metrics
                if m["speculative_num_steps"]
            )
            / total
            if spec_num_steps
            else 0
        )
        aggregate_accept_length = total_completion_tokens / total_spec_verify_ct if total_spec_verify_ct else 0
        spec_success_rate = (
            (aggregate_accept_length - 1) / spec_num_steps if spec_num_steps and total_spec_verify_ct else 0
        )
        spec_output_token_saved_ratio = (
            total_spec_accept_tokens / (total_spec_accept_tokens + total_spec_verify_ct)
            if total_spec_accept_tokens + total_spec_verify_ct
            else 0
        )

        return {
            "total_requests": total,
            "avg_spec_accept_length": round(avg_accept_length, 4),
            "aggregate_spec_accept_length": round(aggregate_accept_length, 4),
            "avg_spec_accept_rate": round(avg_accept_rate, 4),
            "aggregate_spec_accept_rate": round(spec_success_rate, 4),
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

    def save(self, path: str | Path) -> dict[str, Any] | None:
        summary = self.summary()
        if summary is None:
            return None
        path = Path(path)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("SGLang metrics saved to %s", path)
        return summary
