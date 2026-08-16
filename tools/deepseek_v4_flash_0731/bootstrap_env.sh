#!/usr/bin/env bash
# Create an isolated environment by cloning a known-good P800 base environment.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

base_env="${DSV4_BASE_ENV:-}"
[[ -n "${base_env}" ]] || die "set DSV4_BASE_ENV to a working P800 vLLM/PyTorch environment"
require_file "${base_env}/bin/python"

conda_bin="${CONDA_BIN:-$(command -v conda || true)}"
[[ -n "${conda_bin}" ]] || die "conda was not found; set CONDA_BIN explicitly"

mkdir -p "${DSV4_DEPLOY_ROOT}"
if [[ ! -x "${DSV4_ENV_DIR}/bin/python" ]]; then
  "${conda_bin}" create -y -p "${DSV4_ENV_DIR}" --clone "${base_env}"
else
  printf 'Reusing existing environment: %s\n' "${DSV4_ENV_DIR}"
fi

if [[ ! -d "${DSV4_UPSTREAM_DIR}/.git" ]]; then
  git clone --branch v0.25.1 --depth 1 \
    https://github.com/vllm-project/vllm.git "${DSV4_UPSTREAM_DIR}"
fi

python_bin="$(dsv4_python)"

# --no-deps is deliberate: a normal resolver can overwrite Kunlun's custom
# torch/torch_xmlir stack with public CUDA or CPU wheels.
VLLM_TARGET_DEVICE=empty "${python_bin}" -m pip install \
  --no-deps --no-build-isolation -e "${DSV4_UPSTREAM_DIR}"
"${python_bin}" -m pip install \
  --no-deps --no-build-isolation -e "${DSV4_REPO_ROOT}"

"${python_bin}" - <<'PY'
import torch
import compressed_tensors
import huggingface_hub
import safetensors
import torch_plugin
import torch_xmlir
import vllm
import vllm_kunlun

print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("vllm_kunlun", getattr(vllm_kunlun, "__version__", "installed"))
print("compressed_tensors", getattr(compressed_tensors, "__version__", "installed"))
print("huggingface_hub", getattr(huggingface_hub, "__version__", "installed"))
print("safetensors", getattr(safetensors, "__version__", "installed"))
print("Kunlun runtime imports passed")
PY

printf 'Isolated environment ready: %s\n' "${DSV4_ENV_DIR}"
