# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kunlun-specific monkey-patch for ``vllm.v1.worker.block_table``.

Replaces ``BlockTable.compute_slot_mapping`` (which dispatches a Triton
kernel ``_compute_slot_mapping_kernel`` upstream) with the native Kunlun
XPU op ``kunlun_ops.compute_slot_mappings`` when available. Older Kunlun
runtime builds do not export that op, so they use the CPU mirror maintained
by ``CpuGpuBuffer`` and copy the completed mapping back to the device.

Triggering: imported from ``vllm_kunlun.__init__`` post-import hook
once ``vllm.v1.worker.block_table`` is loaded. Idempotent under fork()
and re-import via the ``_kunlun_slot_patched`` flag on the class.
"""

import logging

import kunlun_ops
import numpy as np
import torch
from vllm.v1.worker.block_table import PAD_SLOT_ID
from vllm.v1.worker.block_table import BlockTable as _upstream_cls

logger = logging.getLogger("vllm_kunlun")


def _compute_slot_mapping_native(self, num_reqs, query_start_loc, positions):
    num_tokens = positions.shape[0]
    total_cp_world_size = self.pcp_world_size * self.dcp_world_size
    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
    block_sizes = torch.tensor(
        [self.block_size], dtype=torch.int32, device=self.block_table.gpu.device
    )
    kunlun_ops.compute_slot_mappings(
        [self.slot_mapping.gpu],  # list
        [self.block_table.gpu],  # list
        positions,
        query_start_loc,
        block_sizes,  # int32 tensor
        num_reqs,
        num_tokens,
        PAD_SLOT_ID,
        total_cp_world_size,
        total_cp_rank,
        self.cp_kv_cache_interleave_size,
    )


def _compute_slot_mapping_numpy(self, num_reqs, query_start_loc, positions):
    num_tokens = positions.shape[0]
    max_num_tokens = self.max_num_batched_tokens
    block_size = self.block_size
    block_table_np = self.block_table.np
    slot_mapping_np = self.slot_mapping.np
    total_cp = self.pcp_world_size * self.dcp_world_size

    if total_cp == 1:
        if num_tokens > 0:
            pos_np = positions[:num_tokens].cpu().numpy()
            qsl_np = query_start_loc[: num_reqs + 1].cpu().numpy()
            token_arange = np.arange(num_tokens, dtype=qsl_np.dtype)
            req_idx = np.searchsorted(qsl_np, token_arange, side="right") - 1
            np.clip(req_idx, 0, num_reqs - 1, out=req_idx)
            block_idx = pos_np // block_size
            offset = pos_np - block_idx * block_size
            block_num = block_table_np[req_idx, block_idx].astype(np.int64)
            np.add(
                block_num * block_size,
                offset,
                out=slot_mapping_np[:num_tokens],
            )
        slot_mapping_np[num_tokens:max_num_tokens] = PAD_SLOT_ID
        self.slot_mapping.copy_to_gpu()
        return

    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
    cp_int = self.cp_kv_cache_interleave_size
    virtual_block_size = block_size * total_cp
    qsl_np = query_start_loc[: num_reqs + 1].cpu().numpy()
    pos_np = positions[:num_tokens].cpu().numpy()
    for req_idx in range(num_reqs):
        start = int(qsl_np[req_idx])
        end = int(qsl_np[req_idx + 1])
        if end <= start:
            continue
        pos = pos_np[start:end]
        block_indices = pos // virtual_block_size
        block_numbers = block_table_np[req_idx, block_indices].astype(np.int64)
        virtual_offset = pos - block_indices * virtual_block_size
        is_local = (virtual_offset // cp_int) % total_cp == total_cp_rank
        local_offset = (
            virtual_offset // (total_cp * cp_int)
        ) * cp_int + virtual_offset % cp_int
        slot = block_numbers * block_size + local_offset
        slot_mapping_np[start:end] = np.where(is_local, slot, PAD_SLOT_ID)
    slot_mapping_np[num_tokens:max_num_tokens] = PAD_SLOT_ID
    self.slot_mapping.copy_to_gpu()


def _compute_slot_mapping(self, num_reqs, query_start_loc, positions):
    if hasattr(kunlun_ops, "compute_slot_mappings"):
        return _compute_slot_mapping_native(
            self, num_reqs, query_start_loc, positions
        )
    return _compute_slot_mapping_numpy(
        self, num_reqs, query_start_loc, positions
    )

# Idempotent monkey-patch: safe under fork() and re-import.
if not getattr(_upstream_cls, "_kunlun_slot_patched", False):
    _upstream_cls.compute_slot_mapping = _compute_slot_mapping
    _upstream_cls._kunlun_slot_patched = True
    logger.info(
        "[KunlunPlugin] BlockTable.compute_slot_mapping patched "
        "in vllm_kunlun/v1/worker/block_table.py"
    )
