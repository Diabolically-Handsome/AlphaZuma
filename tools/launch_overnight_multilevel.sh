#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s PREREGISTRATION RUN_ID\n' "$0" >&2
  exit 2
fi

cd "/mnt/c/Users/Laure/Documents/祖玛"
export TMPDIR=/tmp
export TEMP=/tmp
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf 'launcher_utc=%s run_id=%s\n' "$(date --utc --iso-8601=seconds)" "$2"
exec .venv/bin/python tools/train_overnight_multilevel.py \
  --preregistration "$1" \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
  --run-id "$2"
