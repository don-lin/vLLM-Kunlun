"""DeepSeek-V4-Flash-0731 model adapter for vLLM 0.25.1 on Kunlun."""

from __future__ import annotations

import importlib
import sys
import types

from vllm.platforms import current_platform

# ``vllm.models.deepseek_v4`` selects a hardware topology in its package
# initializer. Kunlun is an out-of-tree platform, so upstream would otherwise
# import the NVIDIA topology (and its optional CUDA-only dependencies) before
# this adapter can select the deliberately platform-neutral XPU topology.
_UPSTREAM_PACKAGE = "vllm.models.deepseek_v4"
if _UPSTREAM_PACKAGE not in sys.modules:
    _had_instance_override = "is_xpu" in vars(current_platform)
    _instance_override = vars(current_platform).get("is_xpu")
    _temporary_modules = []

    # Selecting the upstream XPU topology temporarily makes the generic
    # fused-MoE package import Intel's optional vllm_xpu_kernels interface.
    # Kunlun never instantiates that expert backend, but the import happens
    # before our own MoE implementation can be selected.  Provide a scoped
    # placeholder solely while the topology package is initialized.
    if "vllm_xpu_kernels.fused_moe_interface" not in sys.modules:
        _xpu_moe_stub = types.ModuleType("vllm_xpu_kernels.fused_moe_interface")

        class _UnavailableXpuFusedMoe:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "Intel vllm_xpu_kernels is unavailable on Kunlun; "
                    "the Kunlun MoE backend should have been selected"
                )

        _xpu_moe_stub.XpuFusedMoe = _UnavailableXpuFusedMoe
        sys.modules["vllm_xpu_kernels.fused_moe_interface"] = _xpu_moe_stub
        _temporary_modules.append("vllm_xpu_kernels.fused_moe_interface")
    if "vllm_xpu_kernels.flash_attn_interface" not in sys.modules:
        _xpu_attn_stub = types.ModuleType("vllm_xpu_kernels.flash_attn_interface")

        def _unavailable_xpu_flash_attn(*args, **kwargs):
            raise RuntimeError(
                "Intel vllm_xpu_kernels is unavailable on Kunlun; "
                "the Kunlun attention backend should have been selected"
            )

        _xpu_attn_stub.flash_attn_varlen_func = _unavailable_xpu_flash_attn
        sys.modules["vllm_xpu_kernels.flash_attn_interface"] = _xpu_attn_stub
        _temporary_modules.append("vllm_xpu_kernels.flash_attn_interface")

    current_platform.is_xpu = lambda: True
    try:
        importlib.import_module(_UPSTREAM_PACKAGE)
    finally:
        if _had_instance_override:
            current_platform.is_xpu = _instance_override
        else:
            del current_platform.is_xpu
        for _temporary_module in reversed(_temporary_modules):
            sys.modules.pop(_temporary_module, None)

# Import and patch the shared topology only at model-resolution time.  vLLM's
# XPU implementation is the closest platform-neutral 0.25.1 topology (no
# NVIDIA TileLang MegaMoE imports); the objects below replace every XPU-specific
# execution seam used by the main model.
from vllm.models.deepseek_v4 import attention as upstream_attention  # noqa: E402
from vllm.models.deepseek_v4 import sparse_mla as upstream_sparse_mla  # noqa: E402
from vllm.models.deepseek_v4.xpu import model as upstream_model  # noqa: E402
from vllm.v1.attention.backends.mla import indexer as upstream_indexer  # noqa: E402
from vllm.v1.attention.backends.mla import (  # noqa: E402
    sparse_swa as upstream_sparse_swa,
)

from .deepseek_v4_attention import (  # noqa: E402
    KunlunDeepseekCompressor,
    KunlunDeepseekV4Attention,
    KunlunDeepseekV4SparseAttnIndexer,
    fused_indexer_q_rope_quant_reference,
    fused_q_kv_rmsnorm_reference,
    kunlun_build_c128a_topk_metadata,
    kunlun_build_prefill_chunk_metadata,
    kunlun_build_sparse_swa_metadata,
    kunlun_get_compressed_slot_mapping,
)
from .deepseek_v4_mhc import (  # noqa: E402
    KunlunHCHeadOp,
    KunlunMHCFusedPostPreOp,
    KunlunMHCPostOp,
    KunlunMHCPreOp,
)

upstream_attention.DeepseekCompressor = KunlunDeepseekCompressor
upstream_attention.SparseAttnIndexer = KunlunDeepseekV4SparseAttnIndexer
upstream_attention.fused_q_kv_rmsnorm = fused_q_kv_rmsnorm_reference
upstream_attention.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_reference
upstream_indexer.get_compressed_slot_mapping = kunlun_get_compressed_slot_mapping
upstream_indexer.build_prefill_chunk_metadata = kunlun_build_prefill_chunk_metadata
upstream_sparse_mla.get_compressed_slot_mapping = kunlun_get_compressed_slot_mapping
upstream_sparse_mla.build_c128a_topk_metadata = kunlun_build_c128a_topk_metadata
upstream_sparse_swa.DeepseekSparseSWAMetadataBuilder.build = (
    kunlun_build_sparse_swa_metadata
)

upstream_model.DeepseekV4XPUAttention = KunlunDeepseekV4Attention
upstream_model.MHCPreOp = KunlunMHCPreOp
upstream_model.MHCPostOp = KunlunMHCPostOp
upstream_model.MHCFusedPostPreOp = KunlunMHCFusedPostPreOp
upstream_model.HCHeadOp = KunlunHCHeadOp

# Preserve the hash table on RoutedExperts for monolithic W8A8 routing.  In
# upstream 0.25.1 the router owns this tensor, but a monolithic quant method is
# invoked after bypassing the router object and otherwise cannot see it.
_upstream_init_fused_moe_experts = upstream_model.DeepseekV4MoE._init_fused_moe_experts


def _kunlun_init_fused_moe_experts(self, config, quant_config, prefix):
    _upstream_init_fused_moe_experts(self, config, quant_config, prefix)
    routed = getattr(self.experts, "routed_experts", None)
    if routed is not None:
        # Keep this as a runtime reference, not a second registered Parameter;
        # otherwise state_dict/weight loading would expose a checkpoint name
        # that does not exist in the official model.
        object.__setattr__(routed, "_kunlun_hash_indices_table", self.gate.tid2eid)


upstream_model.DeepseekV4MoE._init_fused_moe_experts = _kunlun_init_fused_moe_experts


class KunlunDeepseekV4ForCausalLM(upstream_model.DeepseekV4ForCausalLM):
    """Main-model-only DeepSeek V4 implementation for P800 TP8.

    DSpark and MTP architectures are intentionally not registered.  Runtime
    constraints are checked centrally by ``KunlunPlatform``.
    """


__all__ = ["KunlunDeepseekV4ForCausalLM"]
