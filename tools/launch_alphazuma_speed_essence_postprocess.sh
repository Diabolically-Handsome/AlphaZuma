#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"

stdout="/mnt/d/ZumaTraining/alphazuma-speed-essence-postprocess-s70081401-controller.stdout.log"
stderr="/mnt/d/ZumaTraining/alphazuma-speed-essence-postprocess-s70081401-controller.stderr.log"
if [[ -e "$stdout" || -e "$stderr" ]]; then
  echo "refusing to overwrite controller logs" >&2
  exit 1
fi

export TMPDIR=/tmp
export TEMP=/tmp
export PYTHONPATH="/mnt/c/Users/Laure/Documents/祖玛/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec .venv/bin/python tools/run_alphazuma_speed_essence_postprocess.py \
  --master-preregistration diagnostics/alphazuma-speed-essence-campaign-s70081401-preregistration-v1.json \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
  --output-root /mnt/d/ZumaTraining/alphazuma-speed-essence-postprocess-s70081401-v1 \
  --execution-receipt diagnostics/alphazuma-speed-essence-postprocess-execution-s70081401-receipt-v1.json \
  --poll-seconds 30 \
  >"$stdout" 2>"$stderr"
