#!/usr/bin/env bash
# Collect reproducible host, package, git, process and API state without secrets.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

output_dir="${1:-${DSV4_DEPLOY_ROOT}/diagnostics-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${output_dir}"
python_bin="$(dsv4_python)"

xpu-smi > "${output_dir}/xpu-smi.txt" 2>&1 || true
"${python_bin}" -m pip freeze > "${output_dir}/pip-freeze.txt"
git -C "${DSV4_REPO_ROOT}" status --short --branch > "${output_dir}/git-status.txt"
git -C "${DSV4_REPO_ROOT}" log -n 20 --oneline > "${output_dir}/git-log.txt"
ps -ef > "${output_dir}/processes.txt"
curl --silent --show-error "http://127.0.0.1:${DSV4_PORT}/v1/models" \
  > "${output_dir}/models.json" 2> "${output_dir}/models.error" || true
tail -n 2000 "${DSV4_LOG_FILE}" > "${output_dir}/server-tail.log" 2>/dev/null || true

printf 'Diagnostics written to %s\n' "${output_dir}"
