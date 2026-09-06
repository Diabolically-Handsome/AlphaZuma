#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Laure/Documents/祖玛

export PYTHONUNBUFFERED=1

exec .venv/bin/zuma-train \
  --transfer-training \
  --fidelity-suite /mnt/d/ZumaGolden/diagnostics/fidelity-suite-c200-candidate-v24.json \
  --fidelity-suite-root /mnt/d/ZumaGolden \
  --training-suite /mnt/d/ZumaGolden/training/suites/training-gate-v3-candidate-v9.json \
  --training-suite-root /mnt/d/ZumaGolden \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
  --level Jungle2 \
  --aim-bins 180 \
  --mask-rejected-actions \
  --max-ticks 12000 \
  --max-balls 768 \
  --initial-model /mnt/d/ZumaGolden/training/models/fruit-actor-migrated-v1.zip \
  --reset-optimizer-state \
  --teacher-kl-coef 1000 \
  --total-steps 98304 \
  --num-envs 4 \
  --seed 20260950 \
  --device cuda \
  --rollout-steps 512 \
  --batch-size 512 \
  --ppo-epochs 1 \
  --learning-rate 0.000005 \
  --reward-scale 0.01 \
  --entropy-coef 0 \
  --checkpoint-every 16384 \
  --eval-every 1000000 \
  --eval-episodes 8 \
  --boundary-eval-episodes 32 \
  --eval-num-envs 8 \
  --also-evaluate-stochastic-boundaries \
  --run-dir /mnt/d/ZumaTraining/pc-golden-jungle2-transfer-s20260950-98k-v1
