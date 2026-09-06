#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"
exec .venv/bin/python \
  tools/distill_alphazuma_55_motor_observable_gradual_headonly_v1.py \
  --preregistration \
  diagnostics/alphazuma-55-motor-observable-gradual-headonly-s99081642-preregistration-v1.json \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"
