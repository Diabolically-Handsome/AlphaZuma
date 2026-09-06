#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"
export TMPDIR=/tmp
export TEMP=/tmp
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

printf 'launch_utc=%s\n' "$(date --utc --iso-8601=seconds)"
exec .venv/bin/python tools/train_human_speedrun.py \
  --initial-model /mnt/d/ZumaTraining/pc-golden-jungle2-transfer-s20260950-98k-v1/final_model.zip \
  --run-dir /mnt/d/ZumaTraining/human-speedrun-v1-jungle2-s20261010-2621k-v1 \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
  --level Jungle2 \
  --seed 20261010 \
  --total-steps 2621440 \
  --num-envs 4 \
  --device cuda:0 \
  --rollout-steps 512 \
  --batch-size 512 \
  --ppo-epochs 1 \
  --learning-rate 0.000005 \
  --entropy-coef 0.0 \
  --checkpoint-every 262144 \
  --eval-episodes 0 \
  --max-ticks 12000 \
  --max-balls 768 \
  --reaction-delay-ticks 12 \
  --max-aim-speed-degrees-per-second 1080 \
  --max-aim-acceleration-degrees-per-second-squared 18000 \
  --min-button-interval-ticks 5 \
  --win-reward 10 \
  --failure-reward -10 \
  --time-penalty-per-native-tick -0.0001 \
  --score-progress-reward-cap 0.01
