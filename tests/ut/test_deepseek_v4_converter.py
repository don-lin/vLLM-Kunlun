import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SCRIPT = Path(__file__).parents[2] / "tools" / "convert_deepseek_v4_0731_w8a8.py"
SPEC = importlib.util.spec_from_file_location("dsv4_converter", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(converter)


def test_mxfp4_dequantization_values():
    # Low nibble 1 -> +0.5, high nibble 0xF -> -6, scale exponent 0.
    packed = torch.tensor([[0xF1] * 16], dtype=torch.uint8)
    scales = torch.tensor([[127]], dtype=torch.uint8)
    out = converter.dequantize_mxfp4(packed, scales)
    assert out.shape == (1, 32)
    torch.testing.assert_close(out[0, 0::2], torch.full((16,), 0.5))
    torch.testing.assert_close(out[0, 1::2], torch.full((16,), -6.0))


def test_mxfp4_signed_storage_preserves_nibbles():
    # Official 0731 expert weights use safetensors I8 storage even though the
    # byte contains two unsigned E2M1 nibbles.
    packed = torch.tensor([[-15] * 16], dtype=torch.int8)  # bit pattern 0xF1
    scales = torch.tensor([[127]], dtype=torch.uint8)
    out = converter.dequantize_mxfp4(packed, scales)
    torch.testing.assert_close(out[0, 0::2], torch.full((16,), 0.5))
    torch.testing.assert_close(out[0, 1::2], torch.full((16,), -6.0))


def test_ue8m0_scale_decoding():
    encoded = torch.tensor([115, 127, 130], dtype=torch.uint8)
    decoded = converter.decode_ue8m0_scale(encoded)
    torch.testing.assert_close(decoded, torch.tensor([2.0**-12, 1.0, 8.0]))


def test_block_fp8_dequantizes_ue8m0_scale_bytes():
    weight = torch.tensor([[8.0, -4.0], [2.0, 1.0]])
    # A raw UE8M0 byte of 124 represents 2^-3, not the numeric value 124.
    scale = torch.tensor([[124]], dtype=torch.uint8)
    out = converter.dequantize_block_fp8(weight, scale)
    torch.testing.assert_close(out, weight / 8.0)


def test_channel_int8_roundtrip():
    weight = torch.tensor([[0.0, 1.0, -2.0], [10.0, -5.0, 2.5]])
    quantized, scale = converter.quantize_w8a8_channel(weight)
    assert quantized.dtype == torch.int8
    assert scale.shape == (2, 1)
    reconstructed = quantized.float() * scale
    torch.testing.assert_close(reconstructed, weight, atol=0.08, rtol=0.0)


def test_compressed_tensors_recipe_is_dynamic_w8a8():
    config = converter.quantization_config()
    group = config["config_groups"]["group_0"]
    assert config["format"] == "int-quantized"
    assert group["weights"]["strategy"] == "channel"
    assert group["input_activations"]["dynamic"] is True
    assert group["input_activations"]["strategy"] == "token"


def test_runtime_unquantized_v4_weights_are_preserved():
    preserve = [
        "layers.0.ffn.gate.weight",
        "layers.0.attn.compressor.wkv.weight",
        "layers.0.attn.compressor.wgate.weight",
        "layers.0.attn.indexer.weights_proj.weight",
    ]
    for name in preserve:
        assert converter.should_keep_unquantized(name)
        assert not converter.should_quantize(name, torch.empty(2, 2))
    assert converter.should_quantize("layers.0.attn.wq_b.weight", torch.empty(2, 2))


def test_streaming_conversion_with_cross_shard_scale(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "expert_dtype": "fp4"})
    )

    weight_name = "layers.0.attn.wq_b.weight"
    scale_name = "layers.0.attn.wq_b.scale"
    gate_name = "layers.0.ffn.gate.weight"
    save_file(
        {weight_name: torch.tensor([[0x21] * 16, [0xF1] * 16], dtype=torch.uint8)},
        source / "model-00001-of-00003.safetensors",
    )
    save_file(
        {scale_name: torch.tensor([[127], [127]], dtype=torch.uint8)},
        source / "model-00002-of-00003.safetensors",
    )
    save_file(
        {
            gate_name: torch.arange(8, dtype=torch.bfloat16).view(2, 4),
            "embed.weight": torch.arange(12, dtype=torch.bfloat16).view(3, 4),
        },
        source / "model-00003-of-00003.safetensors",
    )
    weight_map = {
        weight_name: "model-00001-of-00003.safetensors",
        scale_name: "model-00002-of-00003.safetensors",
        gate_name: "model-00003-of-00003.safetensors",
        "embed.weight": "model-00003-of-00003.safetensors",
    }
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )

    stats = converter.convert(source, output)

    converted = load_file(output / "model-00001-of-00003.safetensors")
    output_scale_name = "layers.0.attn.wq_b.weight_scale"
    assert converted[weight_name].dtype == torch.int8
    assert converted[weight_name].shape == (2, 32)
    assert converted[output_scale_name].shape == (2, 1)
    assert not (output / "model-00002-of-00003.safetensors").exists()
    preserved = load_file(output / "model-00003-of-00003.safetensors")
    assert preserved[gate_name].dtype == torch.bfloat16

    config = json.loads((output / "config.json").read_text())
    assert config["expert_dtype"] == "int8"
    assert config["quantization_config"]["quant_method"] == "compressed-tensors"
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert scale_name not in index["weight_map"]
    assert index["weight_map"][output_scale_name] == (
        "model-00001-of-00003.safetensors"
    )
    assert index["metadata"]["total_size"] == stats["output_bytes"]
