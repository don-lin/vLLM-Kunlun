#!/usr/bin/env bash
# Foreground launch command for the only profile exercised on P800 so far.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
configure_p800_runtime

require_file "${DSV4_MODEL_DIR}/config.json"
vllm_bin="${DSV4_ENV_DIR}/bin/vllm"
[[ -x "${vllm_bin}" ]] || die "vllm executable not found: ${vllm_bin}"

exec "${vllm_bin}" serve "${DSV4_MODEL_DIR}" \
  --host "${DSV4_HOST}" \
  --port "${DSV4_PORT}" \
  --served-model-name "${DSV4_SERVED_MODEL_NAME}" \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --block-size 256 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --enforce-eager \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  "$@"
