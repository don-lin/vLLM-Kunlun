#!/usr/bin/env bash
# Start the server in a new process group and wait for /health.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${DSV4_DEPLOY_ROOT}"
if [[ -s "${DSV4_PID_FILE}" ]]; then
  old_pid="$(tr -dc '0-9' < "${DSV4_PID_FILE}")"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    die "server already appears to be running with PID ${old_pid}"
  fi
fi

nohup setsid "${DSV4_TOOLS_DIR}/serve_p800_tp8.sh" "$@" \
  >"${DSV4_LOG_FILE}" 2>&1 </dev/null &
server_pid=$!
printf '%s\n' "${server_pid}" > "${DSV4_PID_FILE}"
printf 'Started PID %s; log: %s\n' "${server_pid}" "${DSV4_LOG_FILE}"

for _ in $(seq 1 "${DSV4_START_TIMEOUT:-900}"); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -n 100 "${DSV4_LOG_FILE}" || true
    die "server exited before becoming healthy"
  fi
  if curl --fail --silent "http://127.0.0.1:${DSV4_PORT}/health" >/dev/null; then
    printf '%s\n' 'Server health endpoint is ready.'
    exit 0
  fi
  sleep 1
done

die "server did not become healthy before timeout; inspect ${DSV4_LOG_FILE}"
