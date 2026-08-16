#!/usr/bin/env bash
# Verify the host, eight P800 devices, Python stack, source tree and free space.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

command -v xpu-smi >/dev/null || die "xpu-smi is not installed or not in PATH"
printf '%s\n' '== XPU inventory =='
xpu-smi

python_bin="$(dsv4_python)"
printf '%s\n' '== Python/XPU runtime =='
"${python_bin}" - <<'PY'
import importlib
import platform

required = (
    "torch",
    "torch_xmlir",
    "torch_plugin",
    "vllm",
    "vllm_kunlun",
    "compressed_tensors",
    "huggingface_hub",
    "safetensors",
)
print("python:", platform.python_version())
for name in required:
    module = importlib.import_module(name)
    print(f"{name}:", getattr(module, "__version__", "installed"))
PY

printf '%s\n' '== Filesystem =='
df -h "${DSV4_DEPLOY_ROOT}"
printf 'source model:    %s\n' "${DSV4_SOURCE_MODEL}"
printf 'converted model: %s\n' "${DSV4_MODEL_DIR}"
printf 'plugin source:   %s\n' "${DSV4_REPO_ROOT}"

device_count="$(xpu-smi 2>/dev/null | grep -c 'P800' || true)"
[[ "${device_count}" -eq 8 ]] || die "expected exactly 8 P800 entries, found ${device_count}"

printf '%s\n' 'Preflight passed.'
