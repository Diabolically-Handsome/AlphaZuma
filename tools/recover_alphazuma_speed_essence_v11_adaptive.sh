#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"

run_root="/mnt/d/ZumaTraining/alphazuma-speed-essence-v11-repair-adaptive-s73081402-v1"
if [[ -e "$run_root" ]]; then
  printf 'refusing recovery because run root exists: %s\n' "$run_root" >&2
  exit 1
fi

stdout_log="${run_root}.recovery-01.stdout.log"
stderr_log="${run_root}.recovery-01.stderr.log"
if [[ -e "$stdout_log" || -e "$stderr_log" ]]; then
  printf 'refusing to overwrite recovery logs\n' >&2
  exit 1
fi

nohup env \
  TMPDIR=/tmp \
  TEMP=/tmp \
  PYTHONPATH=src \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  .venv/bin/python tools/train_overnight_multilevel.py \
    --preregistration diagnostics/alphazuma-speed-essence-v11-repair-s73081401-preregistration-v1.json \
    --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
    --run-id essence-v11-repair-adaptive-5080 \
    >"$stdout_log" \
    2>"$stderr_log" \
    </dev/null &

printf 'recovery_utc=%s pid=%s\n' "$(date --utc --iso-8601=seconds)" "$!"
