#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"
exec .venv/bin/python \
  tools/distill_alphazuma_55_motor_observable_gradual_allaim_v1.py \
  --preregistration \
  diagnostics/alphazuma-55-motor-observable-gradual-allaim-s99081633-preregistration-v1.json \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"
