#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"
export TMPDIR=/tmp
export TEMP=/tmp
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf 'recovery_postprocess_launcher_utc=%s\n' "$(date --utc --iso-8601=seconds)"
exec .venv/bin/python tools/run_overnight_v11_postprocess.py \
  --master-preregistration diagnostics/overnight-v11-multilevel-recovery-s43081301-preregistration-v1.json \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
  --output-root /mnt/d/ZumaTraining/overnight-v11-recovery-postprocess-s43081301-v1 \
  --poll-seconds 60
