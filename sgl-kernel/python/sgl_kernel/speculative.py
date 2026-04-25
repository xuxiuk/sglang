import math

import torch


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor | None = None,
    csd_table_keys: torch.Tensor | None = None,
    csd_delta_pairs: torch.Tensor | None = None,
    csd_delta_counter: torch.Tensor | None = None,
    csd_lookup_hit_ct: torch.Tensor | None = None,
    csd_forced_accept_ct: torch.Tensor | None = None,
    csd_delta_pair_ct: torch.Tensor | None = None,
    csd_table_capacity: int = 0,
    csd_table_max_probe: int = 0,
    csd_delta_capacity: int = 0,
    csd_enabled: bool = False,
    csd_dynamic_update: bool = False,
    csd_force_accept_disabled: bool = False,
    csd_logit_margin: float | None = None,
    csd_prob_ratio: float = 0.01,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    if target_logits is None:
        if csd_enabled:
            raise ValueError("target_logits is required when csd_enabled is True")
        target_logits = target_probs
    if csd_logit_margin is None:
        csd_logit_margin = math.log(csd_prob_ratio)
    if csd_table_keys is None:
        csd_table_keys = torch.empty((1,), dtype=torch.int64, device=target_probs.device)
    if csd_delta_pairs is None:
        csd_delta_pairs = torch.empty((1,), dtype=torch.int64, device=target_probs.device)
    if csd_delta_counter is None:
        csd_delta_counter = torch.zeros((1,), dtype=torch.int32, device=target_probs.device)
    if csd_lookup_hit_ct is None:
        csd_lookup_hit_ct = torch.zeros((1,), dtype=torch.int64, device=target_probs.device)
    if csd_forced_accept_ct is None:
        csd_forced_accept_ct = torch.zeros((1,), dtype=torch.int64, device=target_probs.device)
    if csd_delta_pair_ct is None:
        csd_delta_pair_ct = torch.zeros((1,), dtype=torch.int64, device=target_probs.device)

    torch.ops.sgl_kernel.tree_speculative_sampling_target_only.default(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        target_logits,
        csd_table_keys,
        csd_delta_pairs,
        csd_delta_counter,
        csd_lookup_hit_ct,
        csd_forced_accept_ct,
        csd_delta_pair_ct,
        csd_table_capacity,
        csd_table_max_probe,
        csd_delta_capacity,
        csd_enabled,
        csd_dynamic_update,
        csd_force_accept_disabled,
        csd_logit_margin,
        threshold_single,
        threshold_acc,
        deterministic,
    )


def verify_tree_greedy(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    target_logits: torch.Tensor | None = None,
    csd_table_keys: torch.Tensor | None = None,
    csd_delta_pairs: torch.Tensor | None = None,
    csd_delta_counter: torch.Tensor | None = None,
    csd_lookup_hit_ct: torch.Tensor | None = None,
    csd_forced_accept_ct: torch.Tensor | None = None,
    csd_delta_pair_ct: torch.Tensor | None = None,
    csd_table_capacity: int = 0,
    csd_table_max_probe: int = 0,
    csd_delta_capacity: int = 0,
    csd_enabled: bool = False,
    csd_dynamic_update: bool = False,
    csd_force_accept_disabled: bool = False,
    csd_logit_margin: float | None = None,
    csd_prob_ratio: float = 0.01,
) -> None:
    if target_logits is None:
        if csd_enabled:
            raise ValueError("target_logits is required when csd_enabled is True")
        target_logits = torch.empty(
            (*target_predict.shape, 1), dtype=torch.float32, device=target_predict.device
        )
    if csd_logit_margin is None:
        csd_logit_margin = math.log(csd_prob_ratio)
    if csd_table_keys is None:
        csd_table_keys = torch.empty((1,), dtype=torch.int64, device=target_predict.device)
    if csd_delta_pairs is None:
        csd_delta_pairs = torch.empty((1,), dtype=torch.int64, device=target_predict.device)
    if csd_delta_counter is None:
        csd_delta_counter = torch.zeros((1,), dtype=torch.int32, device=target_predict.device)
    if csd_lookup_hit_ct is None:
        csd_lookup_hit_ct = torch.zeros((1,), dtype=torch.int64, device=target_predict.device)
    if csd_forced_accept_ct is None:
        csd_forced_accept_ct = torch.zeros((1,), dtype=torch.int64, device=target_predict.device)
    if csd_delta_pair_ct is None:
        csd_delta_pair_ct = torch.zeros((1,), dtype=torch.int64, device=target_predict.device)

    torch.ops.sgl_kernel.verify_tree_greedy.default(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        target_logits,
        csd_table_keys,
        csd_delta_pairs,
        csd_delta_counter,
        csd_lookup_hit_ct,
        csd_forced_accept_ct,
        csd_delta_pair_ct,
        csd_table_capacity,
        csd_table_max_probe,
        csd_delta_capacity,
        csd_enabled,
        csd_dynamic_update,
        csd_force_accept_disabled,
        csd_logit_margin,
    )


def build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    torch.ops.sgl_kernel.build_tree_kernel_efficient.default(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )


def reconstruct_indices_from_tree_mask(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    torch.ops.sgl_kernel.reconstruct_indices_from_tree_mask.default(
        tree_mask,
        verified_seq_len,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        batch_size,
        draft_token_num,
    )


def segment_packbits(
    x: torch.Tensor,
    input_indptr: torch.Tensor,
    output_indptr: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> None:
    torch.ops.sgl_kernel.segment_packbits.default(
        x,
        input_indptr,
        output_indptr,
        y,
        batch_size,
        torch.cuda.current_stream().cuda_stream,
    )
