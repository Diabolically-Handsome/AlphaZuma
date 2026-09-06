#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"
exec .venv/bin/python \
  tools/run_alphazuma_55_motor_observable_gradual_headonly_postprocess_v1.py \
  --plan \
  diagnostics/alphazuma-55-motor-observable-gradual-headonly-postprocess-s99081643-plan-v2.json \
  --expected-plan-sha256 \
  sha256:8c1e9ff3c21697256b2d05eff4ad9726f58675a1ce6cb5818539bd0a74bb1261 \
  --poll-seconds 30
