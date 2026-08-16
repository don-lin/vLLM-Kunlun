#!/usr/bin/env bash
# Stop only the process group created by start_server.sh.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

[[ -s "${DSV4_PID_FILE}" ]] || die "PID file not found: ${DSV4_PID_FILE}"
server_pid="$(tr -dc '0-9' < "${DSV4_PID_FILE}")"
[[ -n "${server_pid}" ]] || die "PID file is invalid: ${DSV4_PID_FILE}"

if ! kill -0 "${server_pid}" 2>/dev/null; then
  printf 'PID %s is no longer running. Removing stale PID file.\n' "${server_pid}"
  rm -f "${DSV4_PID_FILE}"
  exit 0
fi

cmdline="$(tr '\0' ' ' < "/proc/${server_pid}/cmdline")"
if [[ "${cmdline}" != *"serve_p800_tp8.sh"* && "${cmdline}" != *"${DSV4_MODEL_DIR}"* ]]; then
  die "PID ${server_pid} does not look like this deployment; refusing to stop it: ${cmdline}"
fi

kill -TERM -- "-${server_pid}"
for _ in $(seq 1 30); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    rm -f "${DSV4_PID_FILE}"
    printf 'Stopped PID %s.\n' "${server_pid}"
    exit 0
  fi
  sleep 1
done

printf 'PID %s did not stop after 30 seconds; sending SIGKILL.\n' "${server_pid}" >&2
kill -KILL -- "-${server_pid}"
rm -f "${DSV4_PID_FILE}"
