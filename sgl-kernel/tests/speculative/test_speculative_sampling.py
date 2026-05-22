import math
import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import tree_speculative_sampling_target_only

CSD_EMPTY_KEY = -1
_UINT64_MASK = (1 << 64) - 1


def _hash64(key: int) -> int:
    x = key & _UINT64_MASK
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
    return (x ^ (x >> 31)) & _UINT64_MASK


def _pack_csd_pair(lhs_token: int, rhs_token: int) -> int:
    return (lhs_token << 32) | rhs_token


def _build_csd_table(keys, capacity: int = 8, max_probe: int = 16):
    table = [CSD_EMPTY_KEY] * capacity
    for key in keys:
        slot = _hash64(key) & (capacity - 1)
        for _ in range(max_probe):
            if table[slot] in (CSD_EMPTY_KEY, key):
                table[slot] = key
                break
            slot = (slot + 1) & (capacity - 1)
        else:
            raise RuntimeError("failed to build CSD test table")
    return torch.tensor(table, dtype=torch.int64, device="cuda")

test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
        [[0, 1, 2, -1], [6, 10, 11, -1]],
        [2, 2],
    ),
]


@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def test_tree_speculative_sampling_target_only(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
):
    """
    Tests the tree_speculative_sampling_target_only function using Pytest parameterization.
    """
    device = "cuda"

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    draft_probs = torch.full_like(target_probs, 0, dtype=torch.float32, device=device)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device).to(torch.float32)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert (
        predicts.tolist() == expected_predicts
    ), f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_index.tolist() == expected_accept_index
    ), f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_token_num.tolist() == expected_accept_token_num
    ), f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"


@pytest.mark.parametrize(
    (
        "draft_logit",
        "csd_enabled",
        "ignore_prob_ratio",
        "expected_predicts",
        "expected_accept_index",
        "expected_accept_token_num",
        "expected_lookup_hit_ct",
        "expected_forced_accept_ct",
        "expected_delta_pair_ct",
    ),
    [
        (9.5, True, False, [3, 2, -1], [[0, 1]], [1], 1, 1, 1),
        (3.0, True, False, [2, -1, -1], [[0, -1]], [0], 1, 0, 0),
        (3.0, True, True, [2, -1, -1], [[0, -1]], [0], 1, 0, 1),
        (9.5, False, False, [2, -1, -1], [[0, -1]], [0], 0, 0, 1),
        (3.0, False, False, [2, -1, -1], [[0, -1]], [0], 0, 0, 0),
        (3.0, False, True, [2, -1, -1], [[0, -1]], [0], 0, 0, 1),
    ],
)
def test_tree_speculative_sampling_target_only_csd_force_accept(
    draft_logit,
    csd_enabled,
    ignore_prob_ratio,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
    expected_lookup_hit_ct,
    expected_forced_accept_ct,
    expected_delta_pair_ct,
):
    device = "cuda"
    candidates = torch.tensor([[0, 3, 4]], dtype=torch.int64, device=device)
    retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    retrive_next_token = torch.tensor([[1, -1, -1]], dtype=torch.int64, device=device)
    retrive_next_sibling = torch.tensor([[-1, -1, -1]], dtype=torch.int64, device=device)

    target_probs = torch.zeros((1, 3, 8), dtype=torch.float32, device=device)
    target_probs[0, 0, 2] = 0.7
    target_probs[0, 0, 3] = 0.2
    target_probs[0, 1, 2] = 1.0
    draft_probs = torch.zeros_like(target_probs)
    target_logits = torch.zeros((1, 3, 8), dtype=torch.float32, device=device)
    target_logits[0, 0, 2] = 10.0
    target_logits[0, 0, 3] = draft_logit

    pair_key = _pack_csd_pair(3, 2)
    csd_table_keys = _build_csd_table([pair_key], capacity=8, max_probe=16)
    csd_delta_pairs = torch.empty((4,), dtype=torch.int64, device=device)
    csd_delta_counter = torch.zeros((1,), dtype=torch.int32, device=device)
    csd_lookup_hit_ct = torch.zeros((1,), dtype=torch.int64, device=device)
    csd_forced_accept_ct = torch.zeros((1,), dtype=torch.int64, device=device)
    csd_delta_pair_ct = torch.zeros((1,), dtype=torch.int64, device=device)

    predicts = torch.full((3,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((1,), dtype=torch.int32, device=device)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=torch.full((1, 3), 0.95, dtype=torch.float32, device=device),
        uniform_samples_for_final_sampling=torch.zeros((1,), dtype=torch.float32, device=device),
        target_probs=target_probs,
        draft_probs=draft_probs,
        target_logits=target_logits,
        csd_table_keys=csd_table_keys,
        csd_delta_pairs=csd_delta_pairs,
        csd_delta_counter=csd_delta_counter,
        csd_lookup_hit_ct=csd_lookup_hit_ct,
        csd_forced_accept_ct=csd_forced_accept_ct,
        csd_delta_pair_ct=csd_delta_pair_ct,
        csd_table_capacity=8,
        csd_table_max_probe=16,
        csd_delta_capacity=4,
        csd_enabled=csd_enabled,
        csd_dynamic_update=True,
        csd_dynamic_update_ignore_prob_ratio=ignore_prob_ratio,
        csd_force_accept_disabled=False,
        csd_logit_margin=math.log(0.5),
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert predicts.tolist() == expected_predicts
    assert accept_index.tolist() == expected_accept_index
    assert accept_token_num.tolist() == expected_accept_token_num
    assert csd_lookup_hit_ct.item() == expected_lookup_hit_ct
    assert csd_forced_accept_ct.item() == expected_forced_accept_ct
    assert csd_delta_pair_ct.item() == expected_delta_pair_ct
    assert csd_delta_counter.item() == expected_delta_pair_ct
    if expected_delta_pair_ct:
        assert csd_delta_pairs[:1].tolist() == [pair_key]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
