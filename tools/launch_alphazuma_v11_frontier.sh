#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"

preregistration="diagnostics/alphazuma-v1.1-frontier-s44081301-preregistration-v1.json"
adaptive_root="/mnt/d/ZumaTraining/alphazuma-v1.1-frontier-adaptive-s44081301-v1"
fixed_root="/mnt/d/ZumaTraining/alphazuma-v1.1-frontier-fixed-s44081302-v1"

for path in "$adaptive_root" "$fixed_root"; do
  if [[ -e "$path" ]]; then
    printf 'refusing to reuse output path: %s\n' "$path" >&2
    exit 1
  fi
done

nohup bash tools/launch_overnight_multilevel.sh \
  "$preregistration" frontier-adaptive-5090 \
  >"${adaptive_root}.launch.stdout.log" \
  2>"${adaptive_root}.launch.stderr.log" \
  </dev/null &
adaptive_pid=$!

nohup bash tools/launch_overnight_multilevel.sh \
  "$preregistration" frontier-fixed-5080 \
  >"${fixed_root}.launch.stdout.log" \
  2>"${fixed_root}.launch.stderr.log" \
  </dev/null &
fixed_pid=$!

printf 'launcher_utc=%s adaptive_pid=%s fixed_pid=%s\n' \
  "$(date --utc --iso-8601=seconds)" "$adaptive_pid" "$fixed_pid"
