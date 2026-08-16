"""Torch-native mHC operators for DeepSeek V4 on Kunlun.

These small adapters have the callable interface used by the upstream XPU
model, but intentionally avoid its Intel-XPU/Triton dispatch.  Eager execution
is required by the platform validation in :mod:`vllm_kunlun.platforms.kunlun`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult, hidden_size = residual.shape[-2:]
    outer_shape = residual.shape[:-2]
    residual_3d = residual.reshape(-1, hc_mult, hidden_size)
    x = residual_3d.flatten(1).float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(x, fn.float())

    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
    pre = pre + hc_pre_eps
    post = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )
    comb = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    comb = (
        torch.softmax(
            comb * hc_scale[2] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult),
            dim=-1,
        )
        + hc_sinkhorn_eps
    )
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(max(0, sinkhorn_repeat - 1)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (pre.unsqueeze(-1) * residual_3d.float()).sum(dim=1)
    return (
        post.view(*outer_shape, hc_mult, 1),
        comb.view(*outer_shape, hc_mult, hc_mult),
        layer_input.to(residual.dtype).view(*outer_shape, hidden_size),
    )


def _mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    mixed = torch.einsum("...ij,...ih->...jh", comb.float(), residual.float())
    return (mixed + post.float() * x.unsqueeze(-2).float()).to(residual.dtype)


class KunlunMHCPreOp:
    def __call__(
        self,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        **_kwargs,
    ):
        return _mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )


class KunlunMHCPostOp:
    def __call__(self, x, residual, post_layer_mix, comb_res_mix, **_kwargs):
        return _mhc_post(x, residual, post_layer_mix, comb_res_mix)


class KunlunMHCFusedPostPreOp:
    def __call__(
        self,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        **_kwargs,
    ):
        residual = _mhc_post(x, residual, post_layer_mix, comb_res_mix)
        post, comb, layer_input = _mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        return residual, post, comb, layer_input


class KunlunHCHeadOp:
    def __call__(self, hidden_states, hc_fn, hc_scale, hc_base, rms_norm_eps, hc_eps):
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs = hidden_states.reshape(-1, hc_mult, hidden_size)
        flat = hs.flatten(1).float()
        flat = flat * torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps
        )
        gates = torch.sigmoid(F.linear(flat, hc_fn.float()) * hc_scale + hc_base)
        gates = gates + hc_eps
        out = (gates.unsqueeze(-1) * hs.float()).sum(dim=1)
        return out.to(hidden_states.dtype).view(*outer_shape, hidden_size)


__all__ = [
    "KunlunHCHeadOp",
    "KunlunMHCFusedPostPreOp",
    "KunlunMHCPostOp",
    "KunlunMHCPreOp",
]
