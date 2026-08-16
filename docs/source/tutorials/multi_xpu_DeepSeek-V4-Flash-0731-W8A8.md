# Multi XPU (DeepSeek-V4-Flash-0731 W8A8)

This is the experimental first-release profile for DeepSeek-V4-Flash-0731 on
eight Kunlun3 P800 cards. It serves the main model only and follows the model
topology and weight mapping in upstream vLLM 0.25.1.

## Supported profile

- Kunlun3 P800 × 8, tensor parallel size 8
- preconverted compressed-tensors W8A8 weights
- BF16 activations and BF16 KV cache
- maximum context length 32768
- eager execution

DSpark/MTP speculative decoding, direct MXFP4 loading, FP8 KV cache, Kunlun
Graph, multi-node serving, and 1M context are not supported by this profile.
The server rejects these combinations early with an actionable error.

## Convert the official checkpoint

The converter processes one safetensors shard at a time. It decodes the
official MXFP4 expert tensors and block-FP8 dense tensors, then writes symmetric
per-output-channel INT8 weights. Use an empty output directory with enough free
space for the converted checkpoint.

```bash
python tools/convert_deepseek_v4_0731_w8a8.py \
  /models/DeepSeek-V4-Flash-0731 \
  /models/DeepSeek-V4-Flash-0731-W8A8
```

You can validate tensor names and formats without writing files first:

```bash
python tools/convert_deepseek_v4_0731_w8a8.py \
  /models/DeepSeek-V4-Flash-0731 \
  /models/DeepSeek-V4-Flash-0731-W8A8 \
  --dry-run
```

The converter updates `config.json` with `expert_dtype: int8` and a
compressed-tensors dynamic-token/per-channel W8A8 recipe. It also writes
`kunlun_w8a8_manifest.json` with conversion counts.

## Serve on P800 × 8

Set the same Kunlun device and communication environment used by other TP8
deployments, then start vLLM with the validated flags:

```bash
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1

vllm serve /models/DeepSeek-V4-Flash-0731-W8A8 \
  --tensor-parallel-size 8 \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --block-size 256 \
  --max-model-len 32768 \
  --enforce-eager \
  --trust-remote-code
```

For accuracy acceptance, compare greedy outputs and task metrics against the
same converted W8A8 checkpoint served by upstream NVIDIA vLLM 0.25.1. Do not
compare the W8A8 result directly with the official MXFP4 checkpoint, because
that mixes backend differences with quantization error.
