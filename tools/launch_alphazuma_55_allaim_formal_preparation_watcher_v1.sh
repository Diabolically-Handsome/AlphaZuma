#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"
exec .venv/bin/python \
  tools/run_alphazuma_55_allaim_formal_preparation_watcher_v1.py \
  --poll-seconds 30
