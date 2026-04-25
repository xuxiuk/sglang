import math
import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import verify_tree_greedy


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


@pytest.mark.parametrize(
    (
        "has_table_entry",
        "draft_logit",
        "force_accept_disabled",
        "expected_predicts",
        "expected_accept_index",
        "expected_accept_token_num",
        "expected_lookup_hit_ct",
        "expected_forced_accept_ct",
    ),
    [
        (True, 9.5, False, [3, 2, -1], [[0, 1, -1]], [1], 1, 1),
        (True, 3.0, False, [2, -1, -1], [[0, -1, -1]], [0], 1, 0),
        (True, 9.5, True, [2, -1, -1], [[0, -1, -1]], [0], 1, 0),
        (False, 9.5, False, [2, -1, -1], [[0, -1, -1]], [0], 0, 0),
    ],
)
def test_verify_tree_greedy_csd(
    has_table_entry,
    draft_logit,
    force_accept_disabled,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
    expected_lookup_hit_ct,
    expected_forced_accept_ct,
):
    candidates = torch.tensor([[0, 3, 4]], dtype=torch.int64, device="cuda")
    retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
    retrive_next_token = torch.tensor([[1, -1, -1]], dtype=torch.int64, device="cuda")
    retrive_next_sibling = torch.tensor([[-1, -1, -1]], dtype=torch.int64, device="cuda")

    target_predict = torch.tensor([[2, 2, 5]], dtype=torch.int64, device="cuda")
    target_logits = torch.zeros((1, 3, 8), dtype=torch.float32, device="cuda")
    target_logits[0, 0, 2] = 10.0
    target_logits[0, 0, 3] = draft_logit
    target_logits[0, 1, 2] = 10.0

    pair_key = _pack_csd_pair(3, 2)
    csd_table_keys = _build_csd_table(
        [pair_key] if has_table_entry else [], capacity=8, max_probe=16
    )
    csd_delta_pairs = torch.empty((4,), dtype=torch.int64, device="cuda")
    csd_delta_counter = torch.zeros((1,), dtype=torch.int32, device="cuda")
    csd_lookup_hit_ct = torch.zeros((1,), dtype=torch.int64, device="cuda")
    csd_forced_accept_ct = torch.zeros((1,), dtype=torch.int64, device="cuda")
    csd_delta_pair_ct = torch.zeros((1,), dtype=torch.int64, device="cuda")

    predicts = torch.full((3,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 3), -1, dtype=torch.int32, device="cuda")
    accept_token_num = torch.zeros((1,), dtype=torch.int32, device="cuda")

    verify_tree_greedy(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
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
        csd_enabled=True,
        csd_dynamic_update=True,
        csd_force_accept_disabled=force_accept_disabled,
        csd_logit_margin=math.log(0.5),
    )

    assert predicts.tolist() == expected_predicts
    assert accept_index.tolist() == expected_accept_index
    assert accept_token_num.tolist() == expected_accept_token_num
    assert csd_lookup_hit_ct.item() == expected_lookup_hit_ct
    assert csd_forced_accept_ct.item() == expected_forced_accept_ct
    assert csd_delta_pair_ct.item() == 1
    assert csd_delta_counter.item() == 1
    assert csd_delta_pairs[:1].tolist() == [pair_key]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
