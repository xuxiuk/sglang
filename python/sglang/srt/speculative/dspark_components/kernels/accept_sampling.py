from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    build_dflash_verify_target_probs,
)
from sglang.srt.speculative.dspark_components.kernels.cap_correct_len import (
    CapCorrectLen,
)
from sglang.srt.speculative.dspark_components.kernels.softmax_temp import SoftmaxTemp
from sglang.srt.speculative.dspark_components.dspark_csd import csd_kernel_kwargs
from sglang.srt.speculative.reject_sampling import chain_speculative_sampling_triton

try:
    from sgl_kernel.speculative import tree_speculative_sampling_target_only
except ImportError:
    tree_speculative_sampling_target_only = None

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_ACCEPT_SAMPLING.get()


class AcceptSampling:
    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        gamma: int,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
        csd_runtime=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_sampling(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
            csd_runtime=csd_runtime,
        )

    @classmethod
    def triton(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        gamma: int,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
        csd_runtime=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_sampling_triton(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
            csd_runtime=csd_runtime,
        )


def _accept_sampling_core(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor],
    csd_runtime=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    device = candidates.device
    if not sampling_info.need_top_k_sampling and not sampling_info.need_top_p_sampling:
        target_probs = SoftmaxTemp.execute(
            logits=target_logits,
            temperatures=sampling_info.temperatures,
            rows_per_request=verify_num_draft_tokens,
        ).view(bs, verify_num_draft_tokens, -1)
    else:
        target_probs = build_dflash_verify_target_probs(
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            draft_token_num=verify_num_draft_tokens,
            bs=bs,
            max_top_k=draft_input.max_top_k,
            uniform_top_k_value=draft_input.uniform_top_k_value,
        )
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    # The shared CUDA tree verifier indexes coins by candidate slot (slot zero
    # is the root), whereas the Triton chain verifier indexes the gamma draft
    # steps densely. Allocate the larger slot-shaped tensor; Triton simply uses
    # its first gamma values.
    coin_cols = (
        verify_num_draft_tokens
        if csd_runtime is not None and csd_runtime.enabled
        else gamma
    )
    uniform_samples = torch.rand((bs, coin_cols), dtype=torch.float32, device=device)
    uniform_samples_final = torch.rand((bs,), dtype=torch.float32, device=device)
    if csd_runtime is not None and csd_runtime.enabled:
        if tree_speculative_sampling_target_only is None:
            raise RuntimeError(
                "DSpark CSD requires the sgl-kernel speculative sampler."
            )
        target_logits_3d = target_logits.reshape(bs, verify_num_draft_tokens, -1).to(
            torch.float32
        )
        tree_speculative_sampling_target_only(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates.to(torch.int64),
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_final,
            target_probs=target_probs,
            draft_probs=draft_probs,
            target_logits=target_logits_3d,
            use_rejection_sampling=True,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
            **csd_kernel_kwargs(csd_runtime, include_entropy=True),
        )
    else:
        chain_speculative_sampling_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_final,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
    correct_len = accept_token_num
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
    else:
        cap_trim_lens = torch.zeros_like(correct_len)
    return correct_len, cap_trim_lens, accept_index, predicts


def accept_sampling(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
    csd_runtime=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    device = candidates.device
    correct_len, cap_trim_lens, accept_index, predicts = _accept_sampling_core(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
        csd_runtime=csd_runtime,
    )
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus, cap_trim_lens


@triton.jit
def _gather_two_level_bonus_kernel(
    accept_index_ptr,
    predicts_ptr,
    correct_len_ptr,
    out_ptr,
    cols,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    cl = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int64)
    accept_pos = tl.load(accept_index_ptr + offs * cols + cl, mask=mask, other=0).to(
        tl.int64
    )
    bonus = tl.load(predicts_ptr + accept_pos, mask=mask, other=0)
    tl.store(out_ptr + offs, bonus.to(tl.int64), mask=mask)


def gather_two_level_bonus_triton(
    *,
    accept_index: torch.Tensor,
    predicts: torch.Tensor,
    correct_len: torch.Tensor,
) -> torch.Tensor:
    bs, cols = accept_index.shape
    accept_index = accept_index.contiguous()
    predicts = predicts.contiguous()
    correct_len = correct_len.contiguous()
    out = torch.empty(bs, dtype=torch.int64, device=accept_index.device)
    BLOCK = 256
    grid = (triton.cdiv(bs, BLOCK),)
    _gather_two_level_bonus_kernel[grid](
        accept_index, predicts, correct_len, out, cols, bs, BLOCK=BLOCK
    )
    return out


def accept_sampling_triton(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
    csd_runtime=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correct_len, cap_trim_lens, accept_index, predicts = _accept_sampling_core(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
        csd_runtime=csd_runtime,
    )
    bonus = gather_two_level_bonus_triton(
        accept_index=accept_index, predicts=predicts, correct_len=correct_len
    )
    return correct_len, bonus, cap_trim_lens
