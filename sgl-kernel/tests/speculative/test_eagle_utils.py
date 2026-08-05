import math
import sys

import pytest
import torch
import torch.nn.functional as F

from sgl_kernel import verify_tree_greedy


def _csd_table(pair: int, capacity: int = 8) -> torch.Tensor:
    mask = (1 << 64) - 1
    value = pair & mask
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & mask
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & mask
    slot = ((value ^ (value >> 31)) & mask) & (capacity - 1)
    table = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
    table[slot] = pair
    return table


def test_verify_tree_greedy():
    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device="cuda")
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10
    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i][j]) < 10:
                target_logits[i][j][18] = 10

    target_predict = torch.argmax(target_logits, dim=-1)
    predict_shape = (12,)

    bs = candidates.shape[0]
    num_spec_step = 4

    predicts = torch.full(
        predict_shape, -1, dtype=torch.int32, device="cuda"
    )  # mutable
    accept_index = torch.full(
        (bs, num_spec_step), -1, dtype=torch.int32, device="cuda"
    )  # mutable
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device="cuda")  # mutable

    verify_tree_greedy(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )

    # Check the expected output.
    assert predicts.tolist() == [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
    assert accept_index.tolist() == [
        [0, 3, 4, 5],
        [6, 10, 11, -1],
    ]
    assert accept_token_num.tolist() == [3, 2]


def test_mtp_greedy_csd_force_accept():
    candidates = torch.tensor([[0, 3, 4]], dtype=torch.int64, device="cuda")
    retrieve_index = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
    retrieve_next = torch.tensor([[1, -1, -1]], dtype=torch.int64, device="cuda")
    retrieve_sibling = torch.full_like(retrieve_next, -1)
    target_predict = torch.tensor([[2, 2, 5]], dtype=torch.int64, device="cuda")
    target_logits = torch.zeros((1, 3, 8), dtype=torch.float32, device="cuda")
    target_logits[0, 0, 2] = 10.0
    target_logits[0, 0, 3] = 9.5
    pair = (3 << 32) | 2
    lookup = torch.zeros((1,), dtype=torch.int64, device="cuda")
    force = torch.zeros_like(lookup)
    delta_count = torch.zeros_like(lookup)
    predicts = torch.full((3,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 3), -1, dtype=torch.int32, device="cuda")
    accept_num = torch.zeros((1,), dtype=torch.int32, device="cuda")

    verify_tree_greedy(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next,
        retrive_next_sibling=retrieve_sibling,
        target_predict=target_predict,
        target_logits=target_logits,
        csd_table_keys=_csd_table(pair),
        csd_delta_pairs=torch.empty((4,), dtype=torch.int64, device="cuda"),
        csd_delta_counter=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        csd_lookup_hit_ct=lookup,
        csd_forced_accept_ct=force,
        csd_delta_pair_ct=delta_count,
        csd_table_capacity=8,
        csd_table_max_probe=16,
        csd_delta_capacity=4,
        csd_enabled=True,
        csd_dynamic_update=True,
        csd_logit_margin=math.log(0.5),
    )

    assert lookup.item() == 1
    assert force.item() == 1
    assert accept_num.item() == 1
    assert predicts[0].item() == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
