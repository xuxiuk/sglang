import math
import sys

import pytest
import torch
import torch.nn.functional as F

from sgl_kernel import tree_speculative_sampling_target_only


def _csd_table(pair: int, capacity: int = 8) -> torch.Tensor:
    mask = (1 << 64) - 1
    value = pair & mask
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & mask
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & mask
    slot = ((value ^ (value >> 31)) & mask) & (capacity - 1)
    table = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
    table[slot] = pair
    return table

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


@pytest.mark.parametrize("draft_logit, expected_force", [(9.5, 1), (3.0, 0)])
def test_mtp_target_only_csd_force_accept(draft_logit, expected_force):
    candidates = torch.tensor([[0, 3, 4]], dtype=torch.int64, device="cuda")
    retrieve_index = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
    retrieve_next = torch.tensor([[1, -1, -1]], dtype=torch.int64, device="cuda")
    retrieve_sibling = torch.full_like(retrieve_next, -1)
    target_probs = torch.zeros((1, 3, 8), dtype=torch.float32, device="cuda")
    target_probs[0, 0, 2] = 0.7
    target_probs[0, 0, 3] = 0.2
    target_probs[0, 1, 2] = 1.0
    target_logits = torch.zeros_like(target_probs)
    target_logits[0, 0, 2] = 10.0
    target_logits[0, 0, 3] = draft_logit
    pair = (3 << 32) | 2
    lookup = torch.zeros((1,), dtype=torch.int64, device="cuda")
    force = torch.zeros_like(lookup)
    delta_count = torch.zeros_like(lookup)
    delta_counter = torch.zeros((1,), dtype=torch.int32, device="cuda")
    predicts = torch.full((3,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
    accept_num = torch.zeros((1,), dtype=torch.int32, device="cuda")

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next,
        retrive_next_sibling=retrieve_sibling,
        uniform_samples=torch.full((1, 3), 0.95, device="cuda"),
        uniform_samples_for_final_sampling=torch.zeros((1,), device="cuda"),
        target_probs=target_probs,
        draft_probs=torch.zeros_like(target_probs),
        target_logits=target_logits,
        csd_table_keys=_csd_table(pair),
        csd_delta_pairs=torch.empty((4,), dtype=torch.int64, device="cuda"),
        csd_delta_counter=delta_counter,
        csd_lookup_hit_ct=lookup,
        csd_forced_accept_ct=force,
        csd_delta_pair_ct=delta_count,
        csd_table_capacity=8,
        csd_table_max_probe=16,
        csd_delta_capacity=4,
        csd_enabled=True,
        csd_dynamic_update=True,
        csd_logit_margin=math.log(0.5),
        deterministic=True,
    )

    assert lookup.item() == 1
    assert force.item() == expected_force
    assert accept_num.item() == expected_force
    assert predicts[0].item() == (3 if expected_force else 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
