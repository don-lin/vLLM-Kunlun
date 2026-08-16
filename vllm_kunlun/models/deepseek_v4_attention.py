"""Kunlun BF16 sparse-MLA implementation for DeepSeek-V4-Flash-0731.

The model topology, cache specifications and metadata builders stay in vLLM
0.25.1.  This module supplies the hardware boundary: torch-native normalization
and compression plus the Kunlun sparse attention path.  The reference fallbacks
are intentionally eager and favor correctness over graph capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)

from vllm_kunlun.ops.deep_gemm import int8_mqa_logits, int8_paged_mqa_logits

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    out = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


def fused_q_kv_rmsnorm_reference(qr, kv, q_weight, kv_weight, eps):
    return rms_norm(qr, q_weight, eps), rms_norm(kv, kv_weight, eps)


def _apply_interleaved_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply GPT-J/interleaved RoPE to the final ``rope_dim`` dimensions."""
    if rope_dim == 0:
        return x
    prefix, rotary = x[..., :-rope_dim], x[..., -rope_dim:]
    cache = cos_sin_cache.index_select(0, positions.long()).float()
    half = rope_dim // 2
    cos, sin = cache[..., :half], cache[..., half : half * 2]
    while cos.ndim < rotary.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    even = rotary[..., 0::2].float()
    odd = rotary[..., 1::2].float()
    if inverse:
        rot_even = even * cos + odd * sin
        rot_odd = odd * cos - even * sin
    else:
        rot_even = even * cos - odd * sin
        rot_odd = odd * cos + even * sin
    rotated = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
    return torch.cat((prefix, rotated.to(x.dtype)), dim=-1)


def fused_indexer_q_rope_quant_reference(
    positions,
    index_q,
    index_q_cos_sin_cache,
    index_weights,
    index_weights_softmax_scale,
    index_weights_head_scale,
    use_fp4=False,
):
    if use_fp4:
        raise NotImplementedError(
            "DeepSeek V4 indexer MXFP4 cache is not supported on Kunlun; "
            "use the default indexer cache format."
        )
    q = _apply_interleaved_rope(
        index_q, positions, index_q_cos_sin_cache, index_q_cos_sin_cache.shape[-1]
    )
    q_flat = q.reshape(-1, q.shape[-1])
    q_int8 = torch.empty_like(q_flat, dtype=torch.int8)
    q_amax = torch.empty((q_flat.shape[0], 1), dtype=torch.float32, device=q.device)
    torch.ops._C.quant2d(q_flat, q_int8, q_amax, force_sdnn=True)
    q_amax = q_amax.view(q.shape[0], q.shape[1])
    weights = index_weights.float() * (q_amax / 127.0)
    weights *= index_weights_softmax_scale * index_weights_head_scale
    return q_int8.view_as(q), weights


def kunlun_get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch implementation of V4's Triton compressed-slot kernel."""
    if out is None:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )
    else:
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    if num_tokens == 0:
        return slot_mapping

    query_lens = torch.diff(query_start_loc).to(torch.long)
    req_ids = torch.repeat_interleave(
        torch.arange(
            query_lens.numel(), device=query_start_loc.device, dtype=torch.long
        ),
        query_lens,
        output_size=num_tokens,
    )
    query_bases = torch.repeat_interleave(
        query_start_loc[:-1].long(), query_lens, output_size=num_tokens
    )
    offsets = torch.arange(num_tokens, device=query_start_loc.device) - query_bases
    positions = seq_lens.long().index_select(0, req_ids)
    positions -= query_lens.index_select(0, req_ids)
    positions += offsets
    compressed_positions = torch.div(positions, compress_ratio, rounding_mode="floor")
    valid = (positions >= 0) & ((positions + 1).remainder(compress_ratio) == 0)
    safe_positions = compressed_positions.clamp(min=0)
    physical_blocks = block_table.long()[
        req_ids, torch.div(safe_positions, block_size, rounding_mode="floor")
    ]
    physical_slots = physical_blocks * block_size + safe_positions.remainder(block_size)
    slot_mapping.copy_(torch.where(valid, physical_slots, -1))
    return slot_mapping


def kunlun_build_c128a_topk_metadata(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    global_decode_buffer: torch.Tensor,
    decode_lens_buffer: torch.Tensor,
    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
):
    """Torch replacement for V4's C128A Triton metadata kernel."""
    num_tokens = positions.shape[0]
    num_prefill_tokens = num_tokens - num_decode_tokens
    global_decode = global_decode_buffer[:num_decode_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens]
    width = min(max_compressed_tokens, global_decode_buffer.shape[-1])
    offsets = torch.arange(width, dtype=torch.long, device=positions.device)

    if num_decode_tokens:
        counts = torch.div(
            positions[:num_decode_tokens].long() + 1,
            compress_ratio,
            rounding_mode="floor",
        ).clamp(max=width)
        req_ids = token_to_req_indices[:num_decode_tokens].long()
        logical = offsets.unsqueeze(0).expand(num_decode_tokens, -1)
        physical_blocks = block_table.long()[
            req_ids.unsqueeze(1), logical // block_size
        ]
        physical = physical_blocks * block_size + logical.remainder(block_size)
        valid = logical < counts.unsqueeze(1)
        valid &= slot_mapping[:num_decode_tokens].unsqueeze(1) >= 0
        global_decode[:, :width].copy_(torch.where(valid, physical, -1).to(torch.int32))
        decode_lens.copy_(
            torch.where(
                slot_mapping[:num_decode_tokens] >= 0,
                counts,
                torch.zeros_like(counts),
            ).to(torch.int32)
        )

    if num_prefill_tokens:
        counts = torch.div(
            positions[num_decode_tokens:].long() + 1,
            compress_ratio,
            rounding_mode="floor",
        ).clamp(max=width)
        logical = offsets.unsqueeze(0).expand(num_prefill_tokens, -1)
        prefill_local[:, :width].copy_(
            torch.where(logical < counts.unsqueeze(1), logical, -1).to(torch.int32)
        )

    return global_decode, decode_lens, prefill_local


def kunlun_build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
):
    """Build V4 prefill metadata without CUDA/Triton kernels.

    The supported P800 profile is TP8 with no decode-context parallelism.
    Eager execution permits the single scalar synchronization used to size the
    gathered compressed-K buffer exactly.
    """
    del compressed_seq_lens_cpu, dcp_rank, cp_kv_cache_interleave_size
    if dcp_world_size != 1:
        raise NotImplementedError(
            "DeepSeek V4 on Kunlun does not support decode-context parallelism."
        )

    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillChunkMetadata,
    )

    device = block_table.device
    compressed_lens = compressed_seq_lens[start_idx:end_idx].to(torch.int32)
    num_reqs = end_idx - start_idx
    cu_seq_lens = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    torch.cumsum(compressed_lens, dim=0, out=cu_seq_lens[1:])
    total_seq_lens = int(cu_seq_lens[-1].item())
    if total_seq_lens == 0:
        return None

    token_to_seq = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        compressed_lens,
        output_size=total_seq_lens,
    )
    local_query_starts = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    ).long()
    query_lens = torch.diff(local_query_starts)
    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    slice_start = 0 if query_slice is None else query_slice.start
    slice_stop = total_query_len if query_slice is None else query_slice.stop
    output_query_len = slice_stop - slice_start

    all_req_ids = torch.repeat_interleave(
        torch.arange(num_reqs, device=device, dtype=torch.long),
        query_lens,
        output_size=total_query_len,
    )
    all_query_bases = torch.repeat_interleave(
        local_query_starts[:-1], query_lens, output_size=total_query_len
    )
    all_offsets = torch.arange(total_query_len, device=device) - all_query_bases
    req_ids = all_req_ids[slice_start:slice_stop]
    offsets = all_offsets[slice_start:slice_stop]
    seq_lens = uncompressed_seq_lens[start_idx:end_idx].long()
    start_positions = seq_lens - query_lens
    causal_lens = torch.div(
        start_positions.index_select(0, req_ids) + offsets + 1,
        compress_ratio,
        rounding_mode="floor",
    ).to(torch.int32)
    cu_seqlen_ks = cu_seq_lens[:-1].index_select(0, req_ids)
    cu_seqlen_ke = cu_seqlen_ks + causal_lens

    token_start = int(query_start_loc_cpu[start_idx].item()) + slice_start
    token_end = token_start + output_query_len
    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather or slice_start > 0,
        local_cu_seq_lens=cu_seq_lens,
        local_total_seq_lens=total_seq_lens,
        max_local_total_seq_lens=total_seq_lens,
    )


class KunlunDeepseekV4SparseAttnIndexer(torch.nn.Module):
    """V4 Lightning Indexer backed by the existing Kunlun INT8 kernels."""

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        if use_fp4_cache:
            raise NotImplementedError(
                "DeepSeek V4 indexer MXFP4 cache is not supported on Kunlun."
            )
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert

    def forward(self, hidden_states, q_quant, k, weights):
        metadata = get_forward_context().attn_metadata
        output = self.topk_indices_buffer
        if output is None:
            raise RuntimeError("DeepSeek V4 requires a top-k index buffer.")
        output[: hidden_states.shape[0]].fill_(-1)
        if not isinstance(metadata, dict):
            return output

        indexer_metadata = metadata[self.k_cache.prefix]
        kv_cache = self.k_cache.kv_cache
        if not self.skip_k_cache_insert:
            torch.ops.xspeedgate_ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                indexer_metadata.slot_mapping,
                self.quant_block_size,
                self.scale_fmt,
            )

        if indexer_metadata.prefill is not None:
            for chunk in indexer_metadata.prefill.chunks:
                self._prefill(q_quant, weights, kv_cache, output, chunk)
        if indexer_metadata.decode is not None:
            self._decode(
                q_quant,
                weights,
                kv_cache,
                output,
                indexer_metadata.decode,
                indexer_metadata.num_decode_tokens,
            )
        return output

    def _prefill(self, q_quant, weights, kv_cache, output, chunk) -> None:
        k_int8 = torch.empty(
            (chunk.total_seq_lens, self.head_dim),
            dtype=torch.int8,
            device=q_quant.device,
        )
        k_scale_bytes = torch.empty(
            (chunk.total_seq_lens, 4), dtype=torch.uint8, device=q_quant.device
        )
        torch.ops.xspeedgate_ops.cp_gather_indexer_k_quant_cache(
            kv_cache=kv_cache,
            dst_k=k_int8,
            dst_scale=k_scale_bytes,
            block_table=chunk.block_table,
            cu_seq_lens=chunk.cu_seq_lens,
        )
        token_slice = slice(chunk.token_start, chunk.token_end)
        query_lens = torch.tensor(
            [0, chunk.token_end - chunk.token_start],
            dtype=torch.int32,
            device=q_quant.device,
        )
        key_lens = torch.tensor(
            [0, chunk.total_seq_lens], dtype=torch.int32, device=q_quant.device
        )
        logits = int8_mqa_logits(
            q_quant[token_slice],
            (k_int8, k_scale_bytes.view(torch.float32)),
            weights[token_slice],
            chunk.cu_seqlen_ks,
            chunk.cu_seqlen_ke,
            query_lens,
            query_lens.cpu(),
            key_lens,
            key_lens.cpu(),
        )
        topk = output[token_slice, : self.topk_tokens]
        torch.ops.xspeedgate_ops.topk_per_row(
            logits=logits,
            srcIndices=topk,
            numRows=logits.shape[0],
            stride0=logits.stride(0),
            stride1=logits.stride(1),
            topK=self.topk_tokens,
            rowStarts=chunk.cu_seqlen_ks,
            rowEnds=chunk.cu_seqlen_ke,
            seqLens=None,
            next_n=None,
        )

    def _decode(
        self, q_quant, weights, kv_cache, output, metadata, num_decode_tokens
    ) -> None:
        # Speculative decode is rejected by the platform, so every request has
        # exactly one query token and no pack/unpack path is required.
        decode_lens = metadata.decode_lens
        if metadata.requires_padding or not torch.all(decode_lens == 1):
            raise RuntimeError(
                "DeepSeek V4 Kunlun indexer only supports non-speculative decode."
            )
        seq_lens = metadata.seq_lens.reshape(-1).to(torch.int32)
        batch_size = seq_lens.numel()
        q_decode = q_quant[:num_decode_tokens].reshape(
            batch_size, 1, *q_quant.shape[1:]
        )
        logits = int8_paged_mqa_logits(
            q_decode,
            kv_cache.unsqueeze(-2),
            weights[:num_decode_tokens],
            seq_lens,
            seq_lens.cpu(),
            metadata.block_table,
            metadata.schedule_metadata,
            max_model_len=self.max_model_len,
        )
        topk = output[:num_decode_tokens, : self.topk_tokens]
        torch.ops.xspeedgate_ops.topk_per_row(
            logits=logits,
            srcIndices=topk,
            numRows=logits.shape[0],
            stride0=logits.stride(0),
            stride1=logits.stride(1),
            topK=self.topk_tokens,
            rowStarts=None,
            rowEnds=None,
            seqLens=seq_lens,
            next_n=1,
        )


def kunlun_build_sparse_swa_metadata(
    self,
    common_prefix_len: int,
    common_attn_metadata,
    fast_build: bool = False,
):
    """Torch metadata builder for the BF16 C4A+SWA attention path."""
    del common_prefix_len, fast_build
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    cm = common_attn_metadata
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(cm, decode_threshold=self.decode_threshold)
    )
    token_to_req = cm.token_to_req_indices(self.token_to_req_indices)
    slot_mapping = cm.slot_mapping
    is_valid = self.is_valid_token[: slot_mapping.shape[0]]
    is_valid.copy_(slot_mapping >= 0)

    decode_indices = self.decode_swa_indices[:num_decode_tokens]
    decode_indices.fill_(-1)
    decode_lens = self.decode_swa_lens[:num_decode_tokens]
    decode_lens.zero_()
    if num_decode_tokens:
        token_ids = torch.arange(
            num_decode_tokens, device=slot_mapping.device, dtype=torch.long
        )
        req_ids = token_to_req[:num_decode_tokens].long()
        query_starts = cm.query_start_loc.long().index_select(0, req_ids)
        query_lens = torch.diff(cm.query_start_loc).long().index_select(0, req_ids)
        seq_lens = cm.seq_lens.long().index_select(0, req_ids)
        positions = seq_lens - query_lens + token_ids - query_starts
        starts = (positions - self.window_size + 1).clamp(min=0)
        lengths = (positions - starts + 1).to(torch.int32)
        lengths = torch.where(
            is_valid[:num_decode_tokens], lengths, torch.zeros_like(lengths)
        )
        decode_lens.copy_(lengths)

        offsets = torch.arange(self.window_size, device=slot_mapping.device)
        logical = starts[:, None] + offsets[None, :]
        valid_entries = offsets[None, :] < lengths[:, None]
        safe_logical = logical.clamp(min=0)
        block_ids = torch.div(safe_logical, self.block_size, rounding_mode="floor")
        physical_blocks = cm.block_table_tensor.long()[req_ids[:, None], block_ids]
        physical = physical_blocks * self.block_size
        physical += safe_logical.remainder(self.block_size)
        decode_indices.view(num_decode_tokens, self.window_size).copy_(
            torch.where(valid_entries, physical, -1).to(torch.int32)
        )

    prefill_seq_lens = None
    prefill_seq_lens_cpu = None
    prefill_query_lens_cpu = None
    prefill_gather_lens = None
    if num_prefills:
        prefill_seq_lens = cm.seq_lens[num_decodes : num_decodes + num_prefills]
        seq_lens_cpu = cm.seq_lens_cpu_upper_bound
        if seq_lens_cpu is None:
            seq_lens_cpu = cm.seq_lens.cpu()
        prefill_seq_lens_cpu = seq_lens_cpu[num_decodes : num_decodes + num_prefills]
        prefill_query_lens_cpu = torch.diff(cm.query_start_loc_cpu)[
            num_decodes : num_decodes + num_prefills
        ].to(torch.int32)
        prefix_lens = prefill_seq_lens - prefill_query_lens_cpu.to(
            prefill_seq_lens.device
        )
        prefill_gather_lens = prefill_query_lens_cpu.to(
            prefill_seq_lens.device
        ) + prefix_lens.clamp(min=0, max=self.window_size - 1)

    return DeepseekSparseSWAMetadata(
        block_table=cm.block_table_tensor,
        slot_mapping=slot_mapping,
        block_size=self.block_size,
        seq_lens=cm.seq_lens,
        query_start_loc=cm.query_start_loc,
        query_start_loc_cpu=cm.query_start_loc_cpu,
        is_valid_token=is_valid,
        token_to_req_indices=token_to_req,
        decode_swa_indices=decode_indices,
        decode_swa_lens=decode_lens,
        prefill_swa_indices=None,
        prefill_swa_lens=None,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=num_prefill_tokens,
        prefill_seq_lens=prefill_seq_lens,
        prefill_seq_lens_cpu=prefill_seq_lens_cpu,
        prefill_gather_lens=prefill_gather_lens,
        prefill_query_lens_cpu=prefill_query_lens_cpu,
        prefill_window_size=self.window_size,
        prefill_max_model_len=self.max_model_len,
        prefill_max_num_batched_tokens=self.max_num_batched_tokens,
        tile_sched_swaonly=None,
        tile_sched_c4a=None,
        tile_sched_c128a=None,
    )


class KunlunDeepseekV4Backend(DeepseekV4FlashMLABackend):
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "KUNLUN_DSV4_MLA_SPARSE"

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return True


def _physical_slots(
    logical: torch.Tensor,
    block_table: torch.Tensor,
    req: int,
    block_size: int,
) -> torch.Tensor:
    logical = logical.long()
    valid = logical >= 0
    safe = logical.clamp(min=0)
    blocks = block_table[req].long().index_select(0, safe // block_size)
    slots = blocks * block_size + safe.remainder(block_size)
    return torch.where(valid, slots, torch.full_like(slots, -1))


def _reference_sparse_attention(
    q: torch.Tensor,
    compressed_cache: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    scale: float,
    sink: torch.Tensor,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Memory-bounded BF16 sparse MQA reference used for eager prefill."""
    result = torch.empty_like(q)
    for start in range(0, q.shape[0], chunk_size):
        end = min(start + chunk_size, q.shape[0])
        gathered_parts = []
        valid_parts = []
        if compressed_cache is not None:
            assert compressed_indices is not None
            rows = compressed_indices[start:end].long()
            valid_parts.append(rows >= 0)
            gathered_parts.append(
                compressed_cache.index_select(0, rows.clamp(min=0).flatten()).view(
                    rows.shape[0], rows.shape[1], -1
                )
            )
        rows = swa_indices[start:end].long()
        valid_parts.append(rows >= 0)
        gathered_parts.append(
            swa_cache.index_select(0, rows.clamp(min=0).flatten()).view(
                rows.shape[0], rows.shape[1], -1
            )
        )
        gathered = torch.cat(gathered_parts, dim=1)
        valid = torch.cat(valid_parts, dim=1)
        scores = torch.einsum("thd,tkd->thk", q[start:end].float(), gathered.float())
        scores.mul_(scale).masked_fill_(~valid.unsqueeze(1), -float("inf"))
        sink_logits = sink[: q.shape[1]].float().view(1, -1, 1)
        probs = torch.softmax(torch.cat((scores, sink_logits), dim=-1), dim=-1)[
            ..., :-1
        ]
        result[start:end] = torch.einsum("thk,tkd->thd", probs, gathered.float()).to(
            q.dtype
        )
    return result


class KunlunDeepseekV4Attention(DeepseekV4Attention):
    backend_cls = KunlunDeepseekV4Backend
    use_fp8_ds_mla_layout = False
    PREFILL_CHUNK_SIZE = 1

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        q = rms_norm(q, None, self.eps)
        q = _apply_interleaved_rope(
            q, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim
        )
        kv = _apply_interleaved_rope(
            kv, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim
        )
        if not isinstance(attn_metadata, dict):
            return q
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", attn_metadata[self.swa_cache_layer.prefix]
        )
        slots = swa_metadata.slot_mapping.long()
        valid = slots >= 0
        if valid.any():
            cache = self.swa_cache_layer.kv_cache.view(-1, self.head_dim)
            cache.index_copy_(0, slots[valid], kv[valid].to(cache.dtype))
        return q

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        o = _apply_interleaved_rope(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.rope_head_dim,
            inverse=True,
        )
        heads_per_group = self.n_local_heads // self.n_local_groups
        grouped = o.view(
            o.shape[0], self.n_local_groups, heads_per_group * self.head_dim
        )

        # wo_a is a grouped projection.  The generic LinearBase forward is a
        # 2-D GEMM, so dequantize its per-channel W8A8 weights and perform the
        # small grouped contraction explicitly.
        weight = self.wo_a.weight
        if weight.dtype == torch.int8:
            scale = self.wo_a.weight_scale.float().reshape(-1)
            # The selected scaled-MM kernel stores INT8 weights as [K, N]
            # after loading, while the grouped einsum below consumes [N, K].
            if weight.shape[0] != scale.numel():
                if weight.shape[1] != scale.numel():
                    raise RuntimeError(
                        "Unexpected DeepSeek V4 wo_a W8A8 weight/scale shape: "
                        f"{tuple(weight.shape)}/{tuple(scale.shape)}"
                    )
                weight = weight.t()
            # KunlunScaledMMLinearKernel converts checkpoint dequant scales
            # (amax/127) to per-channel maxima during post-load processing.
            weight = weight.float() * (scale.reshape(-1, 1) / 127.0)
        weight = weight.view(self.n_local_groups, self.o_lora_rank, -1)
        z = torch.einsum("tgd,grd->tgr", grouped.float(), weight.float())
        z = z.to(o.dtype).flatten(1)
        # wo_b was constructed with return_bias=False and therefore returns a
        # tensor directly (not the conventional ``(output, bias)`` tuple).
        return self.wo_b(z)

    def forward_mqa(self, q, kv, positions, output) -> None:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            output.zero_()
            return
        sparse = cast("DeepseekV4FlashMLAMetadata | None", metadata.get(self.prefix))
        swa = cast("DeepseekSparseSWAMetadata", metadata[self.swa_cache_layer.prefix])
        num_decode = swa.num_decode_tokens
        if num_decode:
            self._decode(q[:num_decode], output[:num_decode], sparse, swa)
        if swa.num_prefill_tokens:
            self._prefill(
                q[num_decode:],
                positions[num_decode:],
                output[num_decode:],
                sparse,
                swa,
            )

    def _decode(self, q, output, sparse, swa) -> None:
        swa_idx = swa.decode_swa_indices
        assert swa_idx is not None
        swa_cache = self.swa_cache_layer.kv_cache.reshape(-1, self.head_dim)
        comp_cache = None
        comp_idx = None
        if self.compress_ratio > 1:
            assert sparse is not None
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                local = self.topk_indices_buffer[: q.shape[0]]
                reqs = swa.token_to_req_indices[: q.shape[0]].cpu().tolist()
                rows = []
                for token in range(q.shape[0]):
                    rows.append(
                        _physical_slots(
                            local[token],
                            sparse.block_table,
                            reqs[token],
                            sparse.block_size // self.compress_ratio,
                        )
                    )
                comp_idx = torch.stack(rows)
            else:
                comp_idx = sparse.c128a_global_decode_topk_indices
                assert comp_idx is not None
                comp_idx = comp_idx.reshape(q.shape[0], -1)
            comp_cache = self.kv_cache.reshape(-1, self.head_dim)

        # Decode is the latency-critical path.  The existing Kunlun BF16 sparse
        # MLA kernel returns its softmax statistics, which lets us add the V4
        # attention sink exactly (the sink has a zero value vector).
        from vllm_kunlun.ops.attention.flashmla import kunlun_flash_mla_with_kvcache

        # Materialize only selected rows. Concatenating the full paged caches
        # would duplicate the entire 32K cache on every layer invocation.
        decode_chunk_size = 16
        flat_swa_idx = swa_idx.reshape(q.shape[0], -1).long()
        for start in range(0, q.shape[0], decode_chunk_size):
            end = min(start + decode_chunk_size, q.shape[0])
            gathered_parts = []
            valid_parts = []
            if comp_cache is not None:
                assert comp_idx is not None
                rows = comp_idx[start:end].long()
                valid_parts.append(rows >= 0)
                gathered_parts.append(
                    comp_cache.index_select(0, rows.clamp(min=0).flatten()).view(
                        rows.shape[0], rows.shape[1], self.head_dim
                    )
                )
            rows = flat_swa_idx[start:end]
            valid_parts.append(rows >= 0)
            gathered_parts.append(
                swa_cache.index_select(0, rows.clamp(min=0).flatten()).view(
                    rows.shape[0], rows.shape[1], self.head_dim
                )
            )
            gathered = torch.cat(gathered_parts, dim=1)
            valid = torch.cat(valid_parts, dim=1)
            batch, width, _ = gathered.shape
            selected_cache = gathered.reshape(batch * width, self.head_dim)
            indices = torch.arange(
                batch * width, device=q.device, dtype=torch.int32
            ).view(batch, width)
            indices.masked_fill_(~valid, -1)
            selected_lens = valid.sum(dim=-1, dtype=torch.int32)
            attn_out, max_logits, p_sums = kunlun_flash_mla_with_kvcache(
                q=q[start:end].unsqueeze(1),
                k_cache=selected_cache,
                cache_seqlens=selected_lens,
                cache_seqlens_cpu=selected_lens.cpu(),
                head_dim_v=self.head_dim,
                softmax_scale=self.scale,
                indices=indices.unsqueeze(1),
                max_seq_kv=selected_cache.shape[0],
            )
            sink_exp = torch.exp(
                self.attn_sink[: q.shape[1]].view(1, 1, -1) - max_logits
            )
            sink_factor = p_sums / (p_sums + sink_exp).clamp_min(1e-20)
            output[start:end].copy_((attn_out * sink_factor.unsqueeze(-1)).squeeze(1))

    def _prefill(self, q, positions, output, sparse, swa) -> None:
        # Build physical sparse rows once on CPU, then execute attention in
        # bounded chunks.  This path is eager by contract and avoids per-token
        # host/device synchronizations.
        token_req = swa.token_to_req_indices[swa.num_decode_tokens :].cpu().tolist()
        pos_cpu = positions.cpu().tolist()
        swa_bt = swa.block_table
        swa_bs = swa.block_size
        comp_cache = None
        if self.compress_ratio > 1:
            assert sparse is not None
            comp_cache = self.kv_cache.reshape(-1, self.head_dim)
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                local_topk = self.topk_indices_buffer[
                    swa.num_decode_tokens : swa.num_decode_tokens + q.shape[0]
                ]
            else:
                local_topk = sparse.c128a_prefill_topk_indices
                assert local_topk is not None

        compressed_rows = []
        swa_rows = []
        for token, (req, pos) in enumerate(zip(token_req, pos_cpu)):
            if comp_cache is not None:
                compressed_rows.append(
                    _physical_slots(
                        local_topk[token],
                        sparse.block_table,
                        req,
                        sparse.block_size // self.compress_ratio,
                    )
                )
            start = max(0, pos - self.window_size + 1)
            logical = torch.arange(start, pos + 1, device=q.device)
            swa_slots = _physical_slots(logical, swa_bt, req, swa_bs)
            if swa_slots.numel() < self.window_size:
                swa_slots = F.pad(
                    swa_slots,
                    (0, self.window_size - swa_slots.numel()),
                    value=-1,
                )
            swa_rows.append(swa_slots[: self.window_size])
        compressed_indices = torch.stack(compressed_rows) if compressed_rows else None
        swa_indices = torch.stack(swa_rows)
        swa_cache = self.swa_cache_layer.kv_cache.reshape(-1, self.head_dim)
        output.copy_(
            _reference_sparse_attention(
                q,
                comp_cache,
                compressed_indices,
                swa_cache,
                swa_indices,
                self.scale,
                self.attn_sink,
            )
        )


class KunlunDeepseekCompressor(DeepseekCompressor):
    """Eager BF16 compressor; indexer K insertion uses the existing Kunlun op."""

    def forward(self, kv_score, positions, rotary_emb) -> None:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return
        kv, score = kv_score.split([self.coff * self.head_dim] * 2, dim=-1)
        state_meta = metadata[self.state_cache.prefix]
        slots = state_meta.slot_mapping.long()
        valid = slots >= 0
        state = self.state_cache.kv_cache.view(-1, self.state_cache.kv_cache.shape[-1])
        if valid.any():
            packed = torch.cat(
                (
                    kv.float(),
                    score.float() + self.ape[positions.remainder(self.compress_ratio)],
                ),
                dim=-1,
            )
            state.index_copy_(0, slots[valid], packed[valid].to(state.dtype))

        boundary = valid & ((positions + 1).remainder(self.compress_ratio) == 0)
        token_ids = boundary.nonzero(as_tuple=False).flatten()
        if token_ids.numel() == 0:
            return
        reqs = state_meta.token_to_req_indices.cpu().tolist()
        pos_list = positions.cpu().tolist()
        outputs = []
        output_tokens = []
        for token in token_ids.cpu().tolist():
            pos = pos_list[token]
            start = pos - self.coff * self.compress_ratio + 1
            logical = torch.arange(max(0, start), pos + 1, device=positions.device)
            physical = _physical_slots(
                logical,
                state_meta.block_table,
                reqs[token],
                state_meta.block_size,
            )
            values = state.index_select(0, physical)
            offset_mask = logical >= (pos - self.compress_ratio + 1)
            head_offset = offset_mask.long() * self.head_dim
            col = torch.arange(self.head_dim, device=values.device)
            gathered_kv = values[:, head_offset[:, None] + col]
            gathered_score = values[
                :,
                self.coff * self.head_dim + head_offset[:, None] + col,
            ]
            compressed = (
                torch.softmax(gathered_score.float(), dim=0) * gathered_kv.float()
            ).sum(dim=0)
            compressed = rms_norm(compressed, self.norm.weight, self.rms_norm_eps)
            compressed_pos = torch.tensor(
                [(pos // self.compress_ratio) * self.compress_ratio],
                dtype=positions.dtype,
                device=positions.device,
            )
            compressed = _apply_interleaved_rope(
                compressed.unsqueeze(0),
                compressed_pos,
                rotary_emb.cos_sin_cache,
                self.rope_head_dim,
            ).squeeze(0)
            outputs.append(compressed)
            output_tokens.append(token)

        values = torch.stack(outputs)
        k_meta = metadata[self.k_cache_prefix]
        out_slots = k_meta.slot_mapping[
            torch.tensor(output_tokens, device=positions.device)
        ].long()
        k_layer = self._static_forward_context[self.k_cache_prefix]
        k_cache = k_layer.kv_cache
        if k_cache.dtype == torch.bfloat16:
            flat = k_cache.view(-1, self.head_dim)
            good = out_slots >= 0
            flat.index_copy_(0, out_slots[good], values[good].to(flat.dtype))
        else:
            # The Lightning Indexer cache remains int8+scale and is consumed by
            # the existing Kunlun sparse-indexer kernels.
            torch.ops.xspeedgate_ops.indexer_k_quant_and_cache(
                values, k_cache, out_slots, 128, "ue8m0"
            )


__all__ = [
    "KunlunDeepseekCompressor",
    "KunlunDeepseekV4SparseAttnIndexer",
    "KunlunDeepseekV4Attention",
    "fused_indexer_q_rope_quant_reference",
    "fused_q_kv_rmsnorm_reference",
    "kunlun_build_prefill_chunk_metadata",
    "kunlun_build_c128a_topk_metadata",
    "kunlun_build_sparse_swa_metadata",
    "kunlun_get_compressed_slot_mapping",
]
