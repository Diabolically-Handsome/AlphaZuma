#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"

export TMPDIR=/tmp
export TEMP=/tmp
export PYTHONPATH="/mnt/c/Users/Laure/Documents/祖玛/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec .venv/bin/python tools/run_alphazuma_v11_frontier_postprocess.py \
  --master-preregistration diagnostics/alphazuma-v1.1-frontier-postprocess-s51081301-preregistration-v1.json \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
  --output-root /mnt/d/ZumaTraining/alphazuma-v1.1-frontier-postprocess-s51081301-v1 \
  --poll-seconds 60
