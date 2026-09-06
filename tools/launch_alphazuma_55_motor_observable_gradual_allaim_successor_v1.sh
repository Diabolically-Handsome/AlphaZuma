#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EXPECTED_MASTER_SHA256" >&2
  exit 2
fi

cd "/mnt/c/Users/Laure/Documents/祖玛"
exec .venv/bin/python \
  tools/run_alphazuma_55_motor_observable_gradual_allaim_successor_v1.py \
  --master \
  diagnostics/alphazuma-55-motor-observable-gradual-allaim-successor-s99081636-preregistration-v1.json \
  --expected-master-sha256 "$1" \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
  --poll-seconds 30
