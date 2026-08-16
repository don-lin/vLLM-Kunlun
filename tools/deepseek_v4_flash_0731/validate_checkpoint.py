#!/usr/bin/env python3
"""Validate the converted checkpoint index, recipe and all weight scales."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-scale", type=float, default=1.0)
    args = parser.parse_args()

    root = args.checkpoint
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())
    manifest = json.loads((root / "kunlun_w8a8_manifest.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"missing shards: {missing}")

    quant = config.get("quantization_config", {})
    if config.get("model_type") != "deepseek_v4":
        raise SystemExit(f"unexpected model_type: {config.get('model_type')!r}")
    if config.get("expert_dtype") != "int8":
        raise SystemExit(f"unexpected expert_dtype: {config.get('expert_dtype')!r}")
    if quant.get("quant_method") != "compressed-tensors":
        raise SystemExit(f"unexpected quantization_config: {quant!r}")

    scale_count = 0
    value_count = 0
    global_min = math.inf
    global_max = -math.inf
    nonfinite = 0
    zeros = 0
    names_by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.endswith(".weight_scale"):
            names_by_shard.setdefault(shard, []).append(name)

    for shard, names in sorted(names_by_shard.items()):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name).float()
                scale_count += 1
                value_count += tensor.numel()
                nonfinite += int((~torch.isfinite(tensor)).sum().item())
                zeros += int((tensor == 0).sum().item())
                global_min = min(global_min, float(tensor.min().item()))
                global_max = max(global_max, float(tensor.max().item()))

    report = {
        "checkpoint": str(root),
        "shards": len(shards),
        "tensors": len(weight_map),
        "scale_tensors": scale_count,
        "scale_values": value_count,
        "scale_min": global_min,
        "scale_max": global_max,
        "scale_zero_values": zeros,
        "scale_nonfinite_values": nonfinite,
        "manifest": manifest,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not scale_count:
        raise SystemExit("no .weight_scale tensors were found")
    if nonfinite or zeros:
        raise SystemExit("invalid zero or non-finite weight scales")
    if global_max > args.max_scale:
        raise SystemExit(
            f"weight scale max {global_max} exceeds limit {args.max_scale}; "
            "the UE8M0 bytes may have been decoded numerically"
        )


if __name__ == "__main__":
    main()
