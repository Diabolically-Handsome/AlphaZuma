#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"

original_root="/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"

launch() {
  local preregistration="$1"
  local run_id="$2"
  local log_root="$3"
  if [[ -e "${log_root}.launcher.stdout.log" || -e "${log_root}.launcher.stderr.log" ]]; then
    printf 'refusing to reuse launcher logs: %s\n' "$log_root" >&2
    return 1
  fi
  nohup env \
    TMPDIR=/tmp \
    TEMP=/tmp \
    PYTHONPATH=src \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    .venv/bin/python tools/train_overnight_multilevel.py \
      --preregistration "$preregistration" \
      --original-root "$original_root" \
      --run-id "$run_id" \
      >"${log_root}.launcher.stdout.log" \
      2>"${log_root}.launcher.stderr.log" \
      </dev/null &
  printf '%s %s\n' "$run_id" "$!"
}

printf 'launcher_utc=%s\n' "$(date --utc --iso-8601=seconds)"

launch \
  diagnostics/alphazuma-speed-essence-v1-lowdrift-s71081401-preregistration-v1.json \
  essence-v1-lowdrift-fixed-5090 \
  /mnt/d/ZumaTraining/alphazuma-speed-essence-v1-lowdrift-fixed-s71081401-v1

launch \
  diagnostics/alphazuma-speed-essence-v1-lowdrift-s71081401-preregistration-v1.json \
  essence-v1-lowdrift-adaptive-5080 \
  /mnt/d/ZumaTraining/alphazuma-speed-essence-v1-lowdrift-adaptive-s71081402-v1

launch \
  diagnostics/alphazuma-speed-essence-v1-speed-s72081401-preregistration-v1.json \
  essence-v1-speed-fixed-5090 \
  /mnt/d/ZumaTraining/alphazuma-speed-essence-v1-speed-fixed-s72081401-v1

launch \
  diagnostics/alphazuma-speed-essence-v1-speed-s72081401-preregistration-v1.json \
  essence-v1-speed-adaptive-5080 \
  /mnt/d/ZumaTraining/alphazuma-speed-essence-v1-speed-adaptive-s72081402-v1

launch \
  diagnostics/alphazuma-speed-essence-v11-repair-s73081401-preregistration-v1.json \
  essence-v11-repair-fixed-5090 \
  /mnt/d/ZumaTraining/alphazuma-speed-essence-v11-repair-fixed-s73081401-v1

launch \
  diagnostics/alphazuma-speed-essence-v11-repair-s73081401-preregistration-v1.json \
  essence-v11-repair-adaptive-5080 \
  /mnt/d/ZumaTraining/alphazuma-speed-essence-v11-repair-adaptive-s73081402-v1
