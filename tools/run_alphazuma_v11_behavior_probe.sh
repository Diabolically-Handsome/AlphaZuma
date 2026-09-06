#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=src

exec nice -n 10 taskset -c 47 \
  .venv/bin/python tools/collect_alphazuma_v11_behavior_probe.py \
  --preregistration diagnostics/alphazuma-v1.1-early-behavior-drift-probe-s52081301-preregistration-v1.json \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"
