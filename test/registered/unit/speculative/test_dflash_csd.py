import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import dflash_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestDFlashCSD(unittest.TestCase):
    def test_table_metadata_rejects_cross_backend_or_block_size(self):
        dflash_utils.validate_dflash_csd_table_metadata(
            {"speculative": {"algorithm": "DFLASH", "num_draft_tokens": 16}},
            block_size=16,
        )
        with self.assertRaisesRegex(ValueError, "DFlash calibration table"):
            dflash_utils.validate_dflash_csd_table_metadata(
                {"speculative": {"algorithm": "EAGLE", "num_draft_tokens": 5}},
                block_size=16,
            )
        with self.assertRaisesRegex(ValueError, "block size mismatch"):
            dflash_utils.validate_dflash_csd_table_metadata(
                {"algorithm": "DFLASH", "block_size": 8}, block_size=16
            )

    def test_greedy_without_csd_keeps_vectorized_path(self):
        with patch.object(
            dflash_utils,
            "verify_tree_greedy",
            side_effect=AssertionError("kernel called"),
        ):
            accept_len, bonus = dflash_utils.compute_dflash_correct_drafts_and_bonus(
                candidates=torch.tensor([[5, 6, 7, 8]], dtype=torch.int64),
                target_predict=torch.tensor([[6, 7, 99, 3]], dtype=torch.int64),
            )

        self.assertEqual(accept_len.tolist(), [2])
        self.assertEqual(bonus.tolist(), [99])

    def test_greedy_csd_dispatches_to_shared_kernel(self):
        captured = {}

        def fake_verify_tree_greedy(**kwargs):
            captured.update(kwargs)
            kwargs["predicts"][:3] = torch.tensor([11, 22, 33], dtype=torch.int32)
            kwargs["accept_index"][0, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
            kwargs["accept_token_num"][0] = 1

        with patch.object(
            dflash_utils, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True
        ), patch.object(
            dflash_utils, "verify_tree_greedy", fake_verify_tree_greedy
        ), patch.object(
            dflash_utils,
            "csd_kernel_kwargs",
            return_value={"csd_enabled": True},
        ):
            accept_len, bonus = dflash_utils.compute_dflash_correct_drafts_and_bonus(
                candidates=torch.tensor([[5, 6, 7]], dtype=torch.int64),
                target_predict=torch.tensor([[6, 7, 8]], dtype=torch.int64),
                next_token_logits=torch.randn(3, 16),
                csd_runtime=SimpleNamespace(enabled=True),
            )

        self.assertEqual(accept_len.tolist(), [1])
        self.assertEqual(bonus.tolist(), [22])
        self.assertTrue(captured["csd_enabled"])
        self.assertEqual(captured["target_logits"].shape, (1, 3, 16))
        self.assertEqual(captured["retrive_next_token"].tolist(), [[1, 2, -1]])

    def test_sampling_csd_forwards_logits_and_runtime(self):
        captured = {}

        def fake_sampling_kernel(**kwargs):
            captured.update(kwargs)
            kwargs["predicts"][:2] = torch.tensor([41, 42], dtype=torch.int32)
            kwargs["accept_index"][0, :2] = torch.tensor([0, 1], dtype=torch.int32)
            kwargs["accept_token_num"][0] = 0

        sampling_info = SimpleNamespace(
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            temperatures=torch.ones((1, 1)),
        )
        with patch.object(
            dflash_utils, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True
        ), patch.object(
            dflash_utils,
            "tree_speculative_sampling_target_only",
            fake_sampling_kernel,
        ), patch.object(
            dflash_utils,
            "csd_kernel_kwargs",
            return_value={
                "csd_enabled": True,
                "csd_force_accept_entropy_threshold": 1.25,
                "csd_force_accept_entropy_min_threshold": 0.75,
            },
        ):
            accept_len, bonus = (
                dflash_utils.compute_dflash_sampling_correct_drafts_and_bonus(
                    candidates=torch.tensor([[5, 6]], dtype=torch.int64),
                    next_token_logits=torch.randn(2, 8),
                    sampling_info=sampling_info,
                    threshold_single=1.0,
                    threshold_acc=1.0,
                    uniform_samples=torch.zeros((1, 2)),
                    uniform_samples_for_final_sampling=torch.zeros((1,)),
                    csd_runtime=SimpleNamespace(enabled=True),
                )
            )

        self.assertEqual(accept_len.tolist(), [0])
        self.assertEqual(bonus.tolist(), [41])
        self.assertTrue(captured["csd_enabled"])
        self.assertEqual(captured["csd_force_accept_entropy_threshold"], 1.25)
        self.assertEqual(
            captured["csd_force_accept_entropy_min_threshold"], 0.75
        )
        self.assertEqual(captured["target_logits"].shape, (1, 2, 8))


if __name__ == "__main__":
    unittest.main()
