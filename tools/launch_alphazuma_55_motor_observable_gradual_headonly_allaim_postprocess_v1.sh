#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EXPECTED_PLAN_SHA256" >&2
  exit 64
fi

cd "/mnt/c/Users/Laure/Documents/祖玛"
exec .venv/bin/python \
  tools/run_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_v1.py \
  --plan \
  diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-postprocess-s99081647-plan-v1.json \
  --expected-plan-sha256 "$1" \
  --poll-seconds 30
