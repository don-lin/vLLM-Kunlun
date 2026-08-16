#!/usr/bin/env bash
# Download all model files concurrently; interrupted runs are resumable.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

python_bin="$(dsv4_python)"
mkdir -p "$(dirname "${DSV4_SOURCE_MODEL}")"
download_args=(
  --repo-id "${DSV4_HF_REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}"
  --output "${DSV4_SOURCE_MODEL}"
  --workers "${DSV4_DOWNLOAD_WORKERS:-32}"
)
if [[ -n "${DSV4_MODEL_REVISION:-}" ]]; then
  download_args+=(--revision "${DSV4_MODEL_REVISION}")
fi
"${python_bin}" "${DSV4_TOOLS_DIR}/download_model.py" "${download_args[@]}"
