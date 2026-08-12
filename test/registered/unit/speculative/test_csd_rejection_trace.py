import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.speculative.csd_rejection_trace import CSDRejectionTraceWriter
from sglang.srt.speculative.csd_runtime import pack_csd_pair


class TestCSDRejectionTrace(unittest.TestCase):
    def test_records_mtp_rejection_and_reconstructable_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = CSDRejectionTraceWriter(Path(directory), capacity=8)
            req = SimpleNamespace(
                rid="req-1",
                origin_input_ids=[10, 11],
                output_ids=[20, 21],
                sampling_params=SimpleNamespace(
                    temperature=1.0, top_p=0.95, top_k=20, max_new_tokens=128
                ),
            )
            batch = SimpleNamespace(reqs=[req])
            # Root/bonus token, one accepted draft, then the rejected draft.
            probs = torch.tensor(
                [
                    [
                        [0.05, 0.05, 0.80, 0.10],
                        [0.10, 0.20, 0.30, 0.40],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                ],
                dtype=torch.float32,
            )
            # Use in-vocabulary candidate IDs for the compact synthetic vocab.
            candidates = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
            logits = probs.clamp_min(1e-6).log()
            predict = torch.tensor([0, 3, 0, 0], dtype=torch.int32)
            accept_index = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)
            table_counts = {pack_csd_pair(2, 3): 9}

            writer.record_mtp_rejections(
                batch=batch,
                candidates=candidates,
                target_probs=probs,
                target_logits=logits,
                predict=predict,
                accept_index=accept_index,
                num_correct_drafts=torch.tensor([1], dtype=torch.int32),
                table_counts=table_counts,
                freq_threshold=6,
                prob_ratio=0.3,
                entropy_min_threshold=-1.0,
                entropy_max_threshold=2.0,
                entropy_basis="target_post_temperature_top_k_top_p",
            )
            writer.flush()

            request = json.loads(writer.requests_path.read_text().strip())
            event = json.loads(writer.events_path.read_text().strip())
            self.assertEqual(request["prompt_token_ids"], [10, 11])
            self.assertEqual(request["generated_token_ids"], [20, 21, 1])
            self.assertEqual(event["generated_position"], 3)
            self.assertEqual(event["draft_token_id"], 2)
            self.assertEqual(event["residual_token_id"], 3)
            self.assertTrue(event["table_hit"])
            self.assertEqual(writer.snapshot()["csd_trace_dropped_ct"], 0)

    def test_capacity_counts_events_not_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = CSDRejectionTraceWriter(Path(directory), capacity=1)
            events = [
                {"request_id": "a"},
                {"request_id": "b"},
            ]
            requests = [
                {"request_id": "a"},
                {"request_id": "b"},
            ]
            writer.submit(requests, events)
            writer.flush()
            snapshot = writer.snapshot()
            self.assertEqual(snapshot["csd_trace_recorded_ct"], 1)
            self.assertEqual(snapshot["csd_trace_dropped_ct"], 1)


if __name__ == "__main__":
    unittest.main()
