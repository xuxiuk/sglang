import math

import pytest
import torch
from sgl_kernel import tree_speculative_sampling_target_only


def _hash64(key: int) -> int:
    mask = (1 << 64) - 1
    value = key & mask
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & mask
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & mask
    return (value ^ (value >> 31)) & mask


def _table_with(key: int, capacity: int = 8) -> torch.Tensor:
    table = torch.full((capacity,), -1, dtype=torch.int64)
    slot = _hash64(key) & (capacity - 1)
    table[slot] = key
    return table.cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "entropy_min_threshold,entropy_threshold,forced",
    [(-1.0, -1.0, 1), (-1.0, 0.1, 0), (0.1, -1.0, 1), (1.0, -1.0, 0)],
)
def test_dspark_csd_preserves_rejection_sampling_and_applies_entropy_gate(
    entropy_min_threshold: float, entropy_threshold: float, forced: int
):
    # Draft token 3 is rejected by classic min(1, p/q): 0.5 * 0.8 > 0.2.
    # Residual sampling picks token 2, so CSD looks up the pair (3, 2).
    device = "cuda"
    candidates = torch.tensor([[0, 3, 4]], dtype=torch.int64, device=device)
    retrieve_index = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    retrieve_next = torch.tensor([[1, -1, -1]], dtype=torch.int64, device=device)
    retrieve_sibling = torch.full_like(retrieve_next, -1)

    target_probs = torch.zeros((1, 3, 8), dtype=torch.float32, device=device)
    target_probs[0, 0, 2] = 0.7
    target_probs[0, 0, 3] = 0.2
    target_probs[0, 0, 4] = 0.1
    target_probs[0, 1, 2] = 1.0
    # DSpark has gamma draft rows but gamma + 1 target/verify rows.  Keep the
    # real layout here so the CUDA kernel's independent batch strides are
    # exercised (batch=2 catches cross-request out-of-bounds indexing).
    candidates = candidates.repeat(2, 1)
    retrieve_index = retrieve_index.repeat(2, 1)
    retrieve_index[1] += 3
    retrieve_next = retrieve_next.repeat(2, 1)
    retrieve_sibling = retrieve_sibling.repeat(2, 1)
    target_probs = target_probs.repeat(2, 1, 1)
    draft_probs = torch.zeros((2, 2, 8), dtype=torch.float32, device=device)
    draft_probs[0, 0, 3] = 0.8
    draft_probs[1, 0, 3] = 0.8
    target_logits = torch.log(target_probs.clamp_min(1e-20))

    pair = (3 << 32) | 2
    table = _table_with(pair)
    delta_pairs = torch.empty((4,), dtype=torch.int64, device=device)
    delta_counter = torch.zeros((1,), dtype=torch.int32, device=device)
    lookup = torch.zeros((1,), dtype=torch.int64, device=device)
    force = torch.zeros((1,), dtype=torch.int64, device=device)
    delta = torch.zeros((1,), dtype=torch.int64, device=device)
    predicts = torch.full((6,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((2, 2), -1, dtype=torch.int32, device=device)
    accept_num = torch.zeros((2,), dtype=torch.int32, device=device)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next,
        retrive_next_sibling=retrieve_sibling,
        uniform_samples=torch.full((2, 3), 0.5, device=device),
        uniform_samples_for_final_sampling=torch.zeros((2,), device=device),
        target_probs=target_probs,
        draft_probs=draft_probs,
        target_logits=target_logits,
        csd_table_keys=table,
        csd_delta_pairs=delta_pairs,
        csd_delta_counter=delta_counter,
        csd_lookup_hit_ct=lookup,
        csd_forced_accept_ct=force,
        csd_delta_pair_ct=delta,
        csd_table_capacity=table.numel(),
        csd_table_max_probe=16,
        csd_delta_capacity=delta_pairs.numel(),
        csd_enabled=True,
        csd_dynamic_update=True,
        csd_dynamic_update_ignore_prob_ratio=True,
        csd_logit_margin=math.log(0.1),
        csd_force_accept_entropy_threshold=entropy_threshold,
        csd_force_accept_entropy_min_threshold=entropy_min_threshold,
        use_rejection_sampling=True,
        deterministic=True,
    )

    assert lookup.item() == 2
    assert force.item() == 2 * forced
    assert delta.item() == 2
    assert delta_counter.item() == 2
    assert delta_pairs[0].item() == pair
    assert delta_pairs[1].item() == pair
    assert accept_num.tolist() == [forced, forced]
    assert predicts[0].item() == (3 if forced else 2)
    assert predicts[3].item() == (3 if forced else 2)
