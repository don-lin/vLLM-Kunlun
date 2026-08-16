#!/usr/bin/env bash
# Convert the official mixed MXFP4/block-FP8 checkpoint to Kunlun W8A8.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

python_bin="$(dsv4_python)"
require_file "${DSV4_SOURCE_MODEL}/config.json"

if [[ -e "${DSV4_MODEL_DIR}" ]] && [[ -n "$(find "${DSV4_MODEL_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  die "output directory is not empty: ${DSV4_MODEL_DIR}"
fi
mkdir -p "${DSV4_MODEL_DIR}"

"${python_bin}" "${DSV4_REPO_ROOT}/tools/convert_deepseek_v4_0731_w8a8.py" \
  "${DSV4_SOURCE_MODEL}" "${DSV4_MODEL_DIR}"
"${python_bin}" "${DSV4_TOOLS_DIR}/validate_checkpoint.py" "${DSV4_MODEL_DIR}"
