#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EXPECTED_PLAN_SHA256" >&2
  exit 2
fi

cd "/mnt/c/Users/Laure/Documents/祖玛"
exec .venv/bin/python \
  tools/run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v2.py \
  --plan \
  diagnostics/alphazuma-55-motor-observable-gradual-allaim-postprocess-s99081634-plan-v2.json \
  --expected-plan-sha256 "$1" \
  --poll-seconds 30
