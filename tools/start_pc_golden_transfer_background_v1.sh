#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s STDOUT_LOG STDERR_LOG\n' "$0" >&2
  exit 64
fi

stdout_log=$1
stderr_log=$2

if [[ -e "$stdout_log" || -e "$stderr_log" ]]; then
  printf 'launch log paths must be absent\n' >&2
  exit 65
fi

nohup /bin/bash tools/launch_pc_golden_transfer_s20260950_v1.sh \
  >"$stdout_log" 2>"$stderr_log" </dev/null &
training_pid=$!
disown "$training_pid" 2>/dev/null || true
printf '%s\n' "$training_pid"
