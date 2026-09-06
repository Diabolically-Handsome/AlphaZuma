#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Laure/Documents/祖玛
export PYTHONPATH="/mnt/c/Users/Laure/Documents/祖玛/tools/inference_optimizer_compat${PYTHONPATH:+:${PYTHONPATH}}"

exec .venv/bin/python \
  tools/run_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_v1.py \
  --plan diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-postprocess-s99081647-plan-v1.json \
  --expected-plan-sha256 sha256:307b328db15dccb14317c5b27a81ebc84b62f16e409bba845dc0ae9012e6af81 \
  --poll-seconds 5
