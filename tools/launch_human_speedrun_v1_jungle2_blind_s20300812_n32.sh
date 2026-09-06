#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"
export TMPDIR=/tmp
export TEMP=/tmp
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

printf 'blind_launch_utc=%s\n' "$(date --utc --iso-8601=seconds)"
exec .venv/bin/python tools/evaluate_human_speedrun_blind.py \
  --preregistration diagnostics/human-speedrun-v1-jungle2-blind-s20300812-n32-preregistration.json \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"
