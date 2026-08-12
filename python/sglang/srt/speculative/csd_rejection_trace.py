from __future__ import annotations

import json
import math
import os
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from sglang.srt.speculative.csd_runtime import pack_csd_pair


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _sampling_params(req) -> dict[str, Any]:
    params = getattr(req, "sampling_params", None)
    if params is None:
        return {}
    names = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "max_new_tokens",
        "seed",
    )
    return {
        name: _json_value(getattr(params, name))
        for name in names
        if hasattr(params, name)
    }


@dataclass
class CSDRejectionTraceWriter:
    output_dir: Path
    capacity: int
    rank: int = 0
    _queue: queue.Queue = field(init=False, repr=False)
    _requests: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _recorded: int = 0
    _dropped: int = 0
    _flushed: int = 0
    _pending: int = 0
    _counter_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / f"events.rank{self.rank}.jsonl"
        self.requests_path = self.output_dir / f"requests.rank{self.rank}.jsonl"
        self.metadata_path = self.output_dir / f"metadata.rank{self.rank}.json"
        for path in (self.events_path, self.requests_path, self.metadata_path):
            if path.exists() and path.stat().st_size:
                raise ValueError(f"CSD rejection trace output already exists: {path}")
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"csd-rejection-trace-rank{self.rank}",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_server_args(cls, server_args, rank: int) -> "CSDRejectionTraceWriter":
        writer = cls(
            Path(server_args.speculative_csd_rejection_trace_dir),
            int(server_args.speculative_csd_rejection_trace_capacity),
            rank,
        )
        metadata = {
            "schema_version": 1,
            "content": "csd_mtp_rejection_trace",
            "rank": rank,
            "pid": os.getpid(),
            "model_path": server_args.model_path,
            "speculative_algorithm": server_args.speculative_algorithm,
            "speculative_num_steps": server_args.speculative_num_steps,
            "speculative_eagle_topk": server_args.speculative_eagle_topk,
            "speculative_num_draft_tokens": server_args.speculative_num_draft_tokens,
            "csd_table_path": server_args.speculative_csd_table_path,
            "csd_freq_threshold": server_args.speculative_csd_freq_threshold,
            "csd_prob_ratio": server_args.speculative_csd_prob_ratio,
            "csd_entropy_min_threshold": server_args.speculative_csd_force_accept_entropy_min_threshold,
            "csd_entropy_max_threshold": server_args.speculative_csd_force_accept_entropy_threshold,
            "force_accept_disabled": server_args.speculative_csd_force_accept_disabled,
            "entropy_basis": "event_specific",
            "entropy_log_base": "e",
        }
        writer.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return writer

    def submit(
        self, requests: list[dict[str, Any]], events: list[dict[str, Any]]
    ) -> None:
        if not events:
            return
        with self._counter_lock:
            available = max(0, self.capacity - self._pending)
            kept_events = events[:available]
            dropped = len(events) - len(kept_events)
            self._dropped += dropped
            if not kept_events:
                return
            kept_rids = {event["request_id"] for event in kept_events}
            kept_requests = [
                request for request in requests if request["request_id"] in kept_rids
            ]
            self._pending += len(kept_events)
            self._recorded += len(kept_events)
        self._queue.put_nowait((kept_requests, kept_events))

    def _write_requests(self) -> None:
        temporary = self.requests_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            for rid in sorted(self._requests):
                file.write(json.dumps(self._requests[rid], ensure_ascii=False) + "\n")
        temporary.replace(self.requests_path)

    def _writer_loop(self) -> None:
        with self.events_path.open("a", encoding="utf-8") as event_file:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break
                requests, events = item
                for request in requests:
                    self._requests[request["request_id"]] = request
                for event in events:
                    event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                event_file.flush()
                with self._counter_lock:
                    self._flushed += len(events)
                    self._pending -= len(events)
                self._write_requests()
                self._queue.task_done()

    def snapshot(self) -> dict[str, int]:
        with self._counter_lock:
            return {
                "csd_trace_recorded_ct": self._recorded,
                "csd_trace_dropped_ct": self._dropped,
                "csd_trace_flushed_ct": self._flushed,
                "csd_trace_buffer_pending": self._pending,
            }

    def flush(self) -> None:
        self._queue.join()
        self._write_requests()

    def record_mtp_rejections(
        self,
        *,
        batch,
        candidates: torch.Tensor,
        target_probs: torch.Tensor,
        target_logits: torch.Tensor,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        table_counts,
        freq_threshold: int,
        prob_ratio: float,
        entropy_min_threshold: float,
        entropy_max_threshold: float,
        entropy_basis: str,
    ) -> None:
        """Record the first rejected token of every topk=1 MTP chain.

        The verifier stops at the first rejection, so there is at most one event
        per request and verify step.  Column zero is the bonus/root token; a
        rejection after k accepted drafts uses candidate k+1 and target row k.
        """
        accepted = num_correct_drafts.detach().to(dtype=torch.int64)
        rows = torch.arange(accepted.numel(), device=accepted.device)
        rejected_cols = accepted + 1
        valid = rejected_cols < candidates.shape[1]
        rows = rows[valid]
        accepted = accepted[valid]
        rejected_cols = rejected_cols[valid]
        if rows.numel() == 0:
            return

        draft_tokens = candidates[rows, rejected_cols].to(dtype=torch.int64)
        last_accepted_indices = accept_index[rows, accepted].to(dtype=torch.int64)
        valid_index = last_accepted_indices >= 0
        rows = rows[valid_index]
        accepted = accepted[valid_index]
        draft_tokens = draft_tokens[valid_index]
        last_accepted_indices = last_accepted_indices[valid_index]
        if rows.numel() == 0:
            return

        residual_tokens = predict[last_accepted_indices].to(dtype=torch.int64)
        distributions = target_probs[rows, accepted].float()
        selected_logits = target_logits[rows, accepted].float()
        draft_probs = distributions.gather(1, draft_tokens[:, None]).squeeze(1)
        residual_probs = distributions.gather(1, residual_tokens[:, None]).squeeze(1)
        entropy = -(
            distributions
            * distributions.clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=1)
        top_values = torch.topk(
            distributions, k=min(2, distributions.shape[1]), dim=1
        ).values
        top1_probs = top_values[:, 0]
        top1_top2_margins = (
            top_values[:, 0] - top_values[:, 1]
            if top_values.shape[1] == 2
            else top_values[:, 0]
        )
        draft_logits = selected_logits.gather(1, draft_tokens[:, None]).squeeze(1)
        max_logits = selected_logits.max(dim=1).values

        # Only compact O(batch) tensors cross the device boundary.  Copying the
        # full vocabulary distribution would make tracing unusable on large
        # vocabularies.
        rows_cpu = rows.cpu().tolist()
        accepted_cpu = accepted.cpu().tolist()
        draft_tokens_cpu = draft_tokens.cpu().tolist()
        residual_tokens_cpu = residual_tokens.cpu().tolist()
        draft_probs_cpu = draft_probs.cpu().tolist()
        residual_probs_cpu = residual_probs.cpu().tolist()
        entropy_cpu = entropy.cpu().tolist()
        top1_probs_cpu = top1_probs.cpu().tolist()
        margins_cpu = top1_top2_margins.cpu().tolist()
        logit_pass_cpu = (
            (draft_logits >= max_logits + math.log(prob_ratio)).cpu().tolist()
        )
        candidates_cpu = candidates.detach().to("cpu", dtype=torch.int64)
        request_records: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        vocab_size = int(target_probs.shape[-1])

        for event_index, row in enumerate(rows_cpu):
            accepted_count = int(accepted_cpu[event_index])
            req = batch.reqs[row]
            rid = str(req.rid)
            output_ids = [int(token) for token in req.output_ids]
            # The worker has not committed this verify step yet.  Include the
            # drafts accepted before the rejection so generated_position is an
            # exact slice boundary in the request-level token stream.
            output_ids.extend(
                int(token)
                for token in candidates_cpu[row, 1 : accepted_count + 1].tolist()
            )
            prompt_ids = [int(token) for token in req.origin_input_ids]
            request_records.append(
                {
                    "request_id": rid,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": output_ids,
                    "sampling_params": _sampling_params(req),
                }
            )
            draft_token = int(draft_tokens_cpu[event_index])
            residual_token = int(residual_tokens_cpu[event_index])
            draft_prob = float(draft_probs_cpu[event_index])
            residual_prob = float(residual_probs_cpu[event_index])
            event_entropy = float(entropy_cpu[event_index])
            pair_key = pack_csd_pair(draft_token, residual_token)
            table_frequency = int(table_counts.get(pair_key, 0))
            table_hit = table_frequency >= freq_threshold
            logit_pass = bool(logit_pass_cpu[event_index])
            entropy_min_pass = (
                entropy_min_threshold < 0 or event_entropy >= entropy_min_threshold
            )
            entropy_max_pass = (
                entropy_max_threshold < 0 or event_entropy <= entropy_max_threshold
            )
            entropy_pass = entropy_min_pass and entropy_max_pass
            events.append(
                {
                    "request_id": rid,
                    "generated_position": len(output_ids),
                    "accepted_drafts_before_rejection": accepted_count,
                    "draft_token_id": draft_token,
                    "residual_token_id": residual_token,
                    "draft_target_probability": draft_prob,
                    "residual_target_probability": residual_prob,
                    "draft_residual_probability_ratio": (
                        draft_prob / residual_prob if residual_prob > 0 else None
                    ),
                    "target_top1_probability": float(top1_probs_cpu[event_index]),
                    "target_top1_top2_margin": float(margins_cpu[event_index]),
                    "entropy_raw": event_entropy,
                    "entropy_norm_vocab": event_entropy / math.log(vocab_size),
                    "effective_support_size": math.exp(event_entropy),
                    "entropy_basis": entropy_basis,
                    "vocab_size": vocab_size,
                    "table_frequency": table_frequency,
                    "table_hit": table_hit,
                    "probability_ratio_gate_passed": logit_pass,
                    "entropy_gate_passed": entropy_pass,
                    "would_force_accept": table_hit and logit_pass and entropy_pass,
                }
            )
        self.submit(request_records, events)
