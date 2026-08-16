#!/usr/bin/env python3
"""Convert DeepSeek-V4-Flash-0731 weights to Kunlun W8A8.

The conversion is shard-streaming: one safetensors shard is materialized at a
time.  It understands the official checkpoint's MXFP4 expert tensors and
block-FP8 dense tensors, then emits compressed-tensors per-output-channel INT8
weights with dynamic per-token INT8 activations.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - exercised by minimal unit-test envs
    safe_open = None
    save_file = None


FP4_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 pairs with UE8M0 scales (32 values per scale byte)."""
    if packed.dtype == torch.int8:
        packed = packed.view(torch.uint8)
    elif packed.dtype != torch.uint8:
        raise TypeError(f"MXFP4 values must be int8/uint8, got {packed.dtype}")
    scale_bytes = scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale
    low = packed.bitwise_and(0x0F).long()
    high = packed.bitwise_right_shift(4).bitwise_and(0x0F).long()
    unpacked = torch.stack((FP4_E2M1[low], FP4_E2M1[high]), dim=-1).flatten(-2)
    blocks = unpacked.reshape(*unpacked.shape[:-1], -1, 32)
    scales = torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0)
    if scales.shape != blocks.shape[:-1]:
        scales = scales.reshape(blocks.shape[:-1])
    return (blocks * scales.unsqueeze(-1)).flatten(-2)


def dequantize_block_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode DeepSeek 128x128 block-FP8 tensors."""
    value = weight.float()
    if scale.numel() == 1:
        return value * scale.float()
    if value.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"Block-FP8 expects 2-D weight/scale, got {value.shape}/{scale.shape}"
        )
    row_block = math.ceil(value.shape[0] / scale.shape[0])
    col_block = math.ceil(value.shape[1] / scale.shape[1])
    expanded = (
        scale.float().repeat_interleave(row_block, 0).repeat_interleave(col_block, 1)
    )
    return value * expanded[: value.shape[0], : value.shape[1]]


def quantize_w8a8_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8 quantization."""
    value = weight.float()
    reduce_dims = tuple(range(1, value.ndim))
    amax = value.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    # compressed-tensors channel scales are [out, 1] for matrix weights.
    return quantized, scale.reshape(value.shape[0], -1)[:, :1].float()


def companion_scale_name(name: str, all_names: set[str]) -> str | None:
    stem = name.removesuffix(".weight")
    for suffix in (".scale", ".weight_scale_inv", ".weight_scale"):
        candidate = stem + suffix
        if candidate in all_names:
            return candidate
    return None


def should_keep_unquantized(name: str) -> bool:
    """Weights whose vLLM V4 modules are constructed with quant_config=None."""
    return (
        name.endswith(".ffn.gate.weight")
        or name.endswith(".compressor.wkv.weight")
        or name.endswith(".compressor.wgate.weight")
        or name.endswith(".indexer.weights_proj.weight")
    )


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or tensor.ndim < 2:
        return False
    return not (
        should_keep_unquantized(name)
        or name.endswith("embed.weight")
        or name.endswith("embed_tokens.weight")
        or name.endswith("head.weight")
        or name.endswith("lm_head.weight")
    )


def quantization_config() -> dict:
    return {
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": None,
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": None,
        "ignore": [
            "lm_head",
            r"re:.*\.ffn\.gate$",
            r"re:.*\.compressor\.fused_wkv_wgate$",
            r"re:.*\.indexer\.weights_proj$",
        ],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.12.2",
    }


class Checkpoint:
    def __init__(self, source: Path):
        if safe_open is None:
            raise RuntimeError("Install safetensors before running this converter.")
        self.source = source
        index_path = source / "model.safetensors.index.json"
        if index_path.exists():
            self.index = json.loads(index_path.read_text())
            self.weight_map: dict[str, str] = self.index["weight_map"]
        else:
            shards = sorted(source.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"No safetensors found in {source}")
            self.weight_map = {}
            for shard in shards:
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    self.weight_map.update({name: shard.name for name in handle.keys()})
            self.index = {"metadata": {}, "weight_map": self.weight_map}
        self._headers: dict[str, tuple[int, dict]] = {}

    def _header(self, shard_name: str) -> tuple[int, dict]:
        cached = self._headers.get(shard_name)
        if cached is not None:
            return cached
        with (self.source / shard_name).open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        cached = (8 + header_size, header)
        self._headers[shard_name] = cached
        return cached

    def load(self, name: str) -> torch.Tensor:
        shard_name = self.weight_map[name]
        data_start, header = self._header(shard_name)
        tensor_info = header[name]
        if tensor_info["dtype"] == "F8_E8M0":
            # PyTorch gained float8_e8m0fnu after the 2.5 frontend shipped
            # with P800.  UE8M0 is exponent-only, so preserving its raw byte
            # representation is exactly what dequantize_mxfp4 expects.
            begin, end = tensor_info["data_offsets"]
            with (self.source / shard_name).open("rb") as handle:
                handle.seek(data_start + begin)
                payload = bytearray(handle.read(end - begin))
            return torch.frombuffer(payload, dtype=torch.uint8).reshape(
                tensor_info["shape"]
            )
        with safe_open(
            self.source / shard_name, framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(name)


def copy_metadata(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if path.suffix in {".safetensors", ".bin"} or path.name.endswith(".index.json"):
            continue
        if path.is_file():
            shutil.copy2(path, output / path.name)


def convert(source: Path, output: Path, dry_run: bool = False) -> dict:
    if save_file is None:
        raise RuntimeError("Install safetensors before running this converter.")
    checkpoint = Checkpoint(source)
    all_names = set(checkpoint.weight_map)
    source_scales = {
        scale
        for name in all_names
        if name.endswith(".weight")
        for scale in [companion_scale_name(name, all_names)]
        if scale is not None
    }
    output_map: dict[str, str] = {}
    stats = {
        "quantized": 0,
        "copied": 0,
        "mxfp4": 0,
        "block_fp8": 0,
        "output_bytes": 0,
    }

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        copy_metadata(source, output)

    shards = sorted(set(checkpoint.weight_map.values()))
    for shard_name in shards:
        names = sorted(
            name for name, shard in checkpoint.weight_map.items() if shard == shard_name
        )
        out_tensors: dict[str, torch.Tensor] = {}
        for name in names:
            if name in source_scales:
                continue
            tensor = checkpoint.load(name)
            if not should_quantize(name, tensor):
                if should_keep_unquantized(name):
                    scale_name = companion_scale_name(name, all_names)
                    if tensor.dtype in (torch.uint8, torch.int8):
                        if scale_name is None:
                            raise ValueError(
                                f"Packed unquantized tensor has no scale: {name}"
                            )
                        tensor = dequantize_mxfp4(
                            tensor, checkpoint.load(scale_name)
                        ).to(torch.bfloat16)
                    elif tensor.dtype.is_floating_point and tensor.dtype not in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.float64,
                    ):
                        if scale_name is None:
                            raise ValueError(
                                f"FP8 unquantized tensor has no scale: {name}"
                            )
                        tensor = dequantize_block_fp8(
                            tensor, checkpoint.load(scale_name)
                        ).to(torch.bfloat16)
                out_tensors[name] = tensor
                output_map[name] = shard_name
                stats["copied"] += 1
                continue

            scale_name = companion_scale_name(name, all_names)
            if tensor.dtype in (torch.uint8, torch.int8):
                if scale_name is None:
                    raise ValueError(f"Packed MXFP4 tensor has no scale: {name}")
                tensor = dequantize_mxfp4(tensor, checkpoint.load(scale_name))
                stats["mxfp4"] += 1
            elif tensor.dtype.is_floating_point and tensor.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ):
                if scale_name is None:
                    raise ValueError(f"FP8 tensor has no block scale: {name}")
                tensor = dequantize_block_fp8(tensor, checkpoint.load(scale_name))
                stats["block_fp8"] += 1

            qweight, weight_scale = quantize_w8a8_channel(tensor)
            scale_out_name = name.removesuffix(".weight") + ".weight_scale"
            out_tensors[name] = qweight
            out_tensors[scale_out_name] = weight_scale
            output_map[name] = shard_name
            output_map[scale_out_name] = shard_name
            stats["quantized"] += 1

        stats["output_bytes"] += sum(
            tensor.numel() * tensor.element_size() for tensor in out_tensors.values()
        )
        if out_tensors and not dry_run:
            save_file(out_tensors, output / shard_name)

    if not dry_run:
        config_path = output / "config.json"
        config = json.loads(config_path.read_text())
        if config.get("model_type") != "deepseek_v4":
            raise ValueError(
                f"Expected model_type=deepseek_v4, got {config.get('model_type')!r}"
            )
        config["expert_dtype"] = "int8"
        config["quantization_config"] = quantization_config()
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        metadata = dict(checkpoint.index.get("metadata", {}))
        metadata["total_size"] = stats["output_bytes"]
        index = {"metadata": metadata, "weight_map": output_map}
        (output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        (output / "kunlun_w8a8_manifest.json").write_text(
            json.dumps({"source": str(source), **stats}, indent=2) + "\n"
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Official DeepSeek-V4-Flash-0731 directory"
    )
    parser.add_argument("output", type=Path, help="Empty output checkpoint directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report without writing"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = convert(args.source.resolve(), args.output.resolve(), args.dry_run)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
