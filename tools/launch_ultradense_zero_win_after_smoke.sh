#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"

smoke_completion="/mnt/d/ZumaTraining/alphazuma-v1.1-ultradense-smoke-5090-8env-e40-s99481301-v1/completion.json"
formal_preregistration="diagnostics/alphazuma-v1.1-ultradense-zero-win-s50081301-preregistration-v1.json"
formal_run_id="ultradense-zero-win-5090-c"
deadline_epoch="$(date --date='2026-08-14T01:30:00Z' +%s)"

printf 'waiter_started_utc=%s\n' "$(date --utc --iso-8601=seconds)"
while [[ ! -f "$smoke_completion" ]]; do
  if (( $(date +%s) >= deadline_epoch )); then
    printf 'smoke completion did not appear before waiter deadline\n' >&2
    exit 1
  fi
  sleep 15
done

.venv/bin/python - "$smoke_completion" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("status") != "COMPLETE":
    raise SystemExit("smoke completion is not COMPLETE")
if int(receipt.get("actual_steps_this_run", -1)) != 131072:
    raise SystemExit("smoke did not complete its frozen 131072-step budget")
PY

printf 'smoke_verified_utc=%s launching=%s\n' \
  "$(date --utc --iso-8601=seconds)" "$formal_run_id"
exec bash tools/launch_overnight_multilevel.sh \
  "$formal_preregistration" "$formal_run_id"
