# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-native logprob helpers for the Kunlun P800 runtime.

vLLM 0.25.1's GPU sampler uses Triton kernels to compute selected-token
logprobs and ranks without materializing a full vocabulary-sized log-softmax.
The P800 Triton runtime accepts those kernels but returns non-finite values.
For the small request batches supported by the current DeepSeek V4 profile,
native torch reductions are both reliable and sufficiently memory bounded.
"""

from __future__ import annotations

import torch
from vllm.v1.outputs import LogprobsTensors


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Compute selected-token logprobs using stable FP32 torch reductions."""
    logits_fp32 = logits.float()
    token_ids = token_ids.to(torch.int64)
    selected = logits_fp32.gather(1, token_ids)
    return selected - torch.logsumexp(logits_fp32, dim=-1, keepdim=True)


compute_token_logprobs._kunlun_patched = True


def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
    logprob_token_ids_state=None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
) -> LogprobsTensors:
    """Kunlun equivalent of vLLM's Triton ``compute_topk_logprobs``."""
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape
    sampled_token_ids = sampled_token_ids.to(torch.int64)

    if max_per_req_token_ids == 0:
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat(
                (logprob_token_ids, topk_indices), dim=1
            )
        logprobs = compute_token_logprobs(logits, logprob_token_ids)
    else:
        if logprob_token_ids_state is None or expanded_idx_mapping is None:
            raise RuntimeError(
                "Kunlun logprob_token_ids require request state and index mapping."
            )
        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
        else:
            topk_token_ids = sampled_token_ids.new_empty((batch_size, 0))

        num_cols = max(num_logprobs, max_per_req_token_ids)
        logprob_token_ids = sampled_token_ids.new_zeros(
            (batch_size, 1 + num_cols)
        )
        valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
        logprob_token_ids[:, 0] = sampled_token_ids
        valid_mask[:, 0] = True

        req_indices = expanded_idx_mapping.to(torch.int64)
        num_custom = (
            logprob_token_ids_state.num_token_ids.gpu.index_select(
                0, req_indices
            )
            .to(torch.int64)
        )
        custom_ids = logprob_token_ids_state.token_ids.gpu.index_select(
            0, req_indices
        )
        for row in range(batch_size):
            custom_count = int(num_custom[row].item())
            if custom_count > 0:
                width = min(custom_count, num_cols)
                logprob_token_ids[row, 1 : 1 + width] = custom_ids[row, :width]
            else:
                width = min(num_logprobs, num_cols)
                if width:
                    logprob_token_ids[row, 1 : 1 + width] = topk_token_ids[
                        row, :width
                    ]
            valid_mask[row, 1 : 1 + width] = True

        logprobs = compute_token_logprobs(logits, logprob_token_ids)
        logprobs = logprobs.masked_fill(~valid_mask, float("-inf"))

    sampled_logits = logits.float().gather(
        1, sampled_token_ids.unsqueeze(-1)
    )
    token_ranks = (logits.float() >= sampled_logits).sum(dim=-1, dtype=torch.int64)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )


compute_topk_logprobs._kunlun_patched = True


__all__ = ["compute_token_logprobs", "compute_topk_logprobs"]
