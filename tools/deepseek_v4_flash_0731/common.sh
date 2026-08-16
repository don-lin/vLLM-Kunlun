#!/usr/bin/env bash
# Shared paths and the validated P800 runtime environment.

set -euo pipefail

DSV4_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSV4_REPO_ROOT="$(cd "${DSV4_TOOLS_DIR}/../.." && pwd)"

export DSV4_DEPLOY_ROOT="${DSV4_DEPLOY_ROOT:-/root/vllm_p800/dsv4}"
export DSV4_ENV_DIR="${DSV4_ENV_DIR:-${DSV4_DEPLOY_ROOT}/conda}"
export DSV4_UPSTREAM_DIR="${DSV4_UPSTREAM_DIR:-${DSV4_DEPLOY_ROOT}/upstream-vllm-0.25.1}"
export DSV4_SOURCE_MODEL="${DSV4_SOURCE_MODEL:-${DSV4_DEPLOY_ROOT}/models/DeepSeek-V4-Flash-0731}"
export DSV4_MODEL_DIR="${DSV4_MODEL_DIR:-${DSV4_DEPLOY_ROOT}/models/DeepSeek-V4-Flash-0731-W8A8}"
export DSV4_SERVED_MODEL_NAME="${DSV4_SERVED_MODEL_NAME:-deepseek-v4-flash-0731}"
export DSV4_HOST="${DSV4_HOST:-0.0.0.0}"
export DSV4_PORT="${DSV4_PORT:-8000}"
export DSV4_LOG_FILE="${DSV4_LOG_FILE:-${DSV4_DEPLOY_ROOT}/server.log}"
export DSV4_PID_FILE="${DSV4_PID_FILE:-${DSV4_DEPLOY_ROOT}/server.pid}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -e "$1" ]] || die "required path does not exist: $1"
}

dsv4_python() {
  local python_bin="${DSV4_ENV_DIR}/bin/python"
  [[ -x "${python_bin}" ]] || die "Python environment not found: ${python_bin}"
  printf '%s\n' "${python_bin}"
}

configure_p800_runtime() {
  local python_bin site_dir
  python_bin="$(dsv4_python)"
  site_dir="$(${python_bin} -c 'import site; print(site.getsitepackages()[0])')"

  export XPU_VISIBLE_DEVICES="${XPU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export XFT_USE_FAST_SWIGLU="${XFT_USE_FAST_SWIGLU:-1}"
  export XPU_USE_FAST_SWIGLU="${XPU_USE_FAST_SWIGLU:-1}"
  export XMLIR_CUDNN_ENABLED="${XMLIR_CUDNN_ENABLED:-1}"
  export XPU_USE_DEFAULT_CTX="${XPU_USE_DEFAULT_CTX:-1}"
  export XMLIR_FORCE_USE_XPU_GRAPH="${XMLIR_FORCE_USE_XPU_GRAPH:-0}"
  export XPU_USE_MOE_SORTED_THRES="${XPU_USE_MOE_SORTED_THRES:-128}"
  export XMLIR_ENABLE_MOCK_TORCH_COMPILE="${XMLIR_ENABLE_MOCK_TORCH_COMPILE:-false}"
  export USE_ORI_ROPE="${USE_ORI_ROPE:-1}"
  export VLLM_HOST_IP="${VLLM_HOST_IP:-$(hostname -I | awk '{print $1}')}"

  # Do not inherit 0.15.x-era switches or debugging knobs.
  unset VLLM_USE_V1 XPU_DUMMY_EVENT CUDA_LAUNCH_BLOCKING

  export LD_LIBRARY_PATH="${DSV4_ENV_DIR}/lib:${DSV4_ENV_DIR}/lib64:${DSV4_ENV_DIR}/xcudart/lib:${site_dir}/xcudart/lib:${site_dir}/torch_xmlir:${site_dir}/torch_xmlir/lib:${site_dir}/torch/lib:${LD_LIBRARY_PATH:-}"
}
