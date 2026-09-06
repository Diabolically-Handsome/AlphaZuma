#!/usr/bin/env bash
set -euo pipefail

project_root="/mnt/c/Users/Laure/Documents/祖玛"
ultradense_checkpoints="/mnt/d/ZumaTraining/alphazuma-v1.1-ultradense-zero-win-5090-c-s50081301-v1/checkpoints"
output="/mnt/d/ZumaTraining/alphazuma-v1.1-checkpoint-recovery-audit-7of7-s53081301-v1/audit.json"
deadline_epoch=$(date -u -d "2026-08-14T02:00:00Z" +%s)

if [[ -e "${output}" ]]; then
  echo "refusing to overwrite existing audit: ${output}" >&2
  exit 1
fi

while true; do
  now_epoch=$(date -u +%s)
  if (( now_epoch >= deadline_epoch )); then
    echo "timed out waiting for a stable ultradense checkpoint" >&2
    exit 124
  fi
  stable_checkpoint=""
  shopt -s nullglob
  for checkpoint in "${ultradense_checkpoints}"/*.zip; do
    modified_epoch=$(stat -c %Y "${checkpoint}")
    if (( now_epoch - modified_epoch >= 45 )); then
      stable_checkpoint="${checkpoint}"
      break
    fi
  done
  shopt -u nullglob
  if [[ -n "${stable_checkpoint}" ]]; then
    printf 'stable checkpoint detected: %s\n' "${stable_checkpoint}"
    break
  fi
  printf 'waiting at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 30
done

cd "${project_root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=src

exec nice -n 10 taskset -c 47 \
  .venv/bin/python tools/audit_alphazuma_v11_checkpoints.py \
  --master-preregistration diagnostics/alphazuma-v1.1-frontier-postprocess-s51081301-preregistration-v1.json \
  --output "${output}" \
  --stable-age-seconds 45 \
  --device cpu
