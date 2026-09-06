#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"

TRAINING_WINDOWS_PID=41460
HEADONLY_TRAINER_PID=401221
TRAINER_NAME="distill_alphazuma_55_motor_observable_gradual_headonly_allaim_v1.py"
RUN_ROOT="/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-v1"
TRAINING_STDOUT="/mnt/d/ZumaTraining/alphazuma-55-headonly-allaim-s99081646-watcher.stdout.log"
TRAINING_STDERR="/mnt/d/ZumaTraining/alphazuma-55-headonly-allaim-s99081646-watcher.stderr.log"
LAUNCH_RECEIPT="diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-launch-receipt-v1.json"
PLAN="diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-postprocess-s99081647-plan-v1.json"
PREREG_AUDIT="diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-postprocess-s99081647-independent-prereg-audit-v1.json"
MASTER="diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-successor-s99081649-preregistration-v1.json"
REGISTRY="diagnostics/alphazuma-55-weekend-formal-seed-registry-s99081650-v10.json"
POST_STDOUT="/mnt/d/ZumaTraining/alphazuma-55-headonly-allaim-s99081647-postprocess.stdout.log"
POST_STDERR="/mnt/d/ZumaTraining/alphazuma-55-headonly-allaim-s99081647-postprocess.stderr.log"
SUCCESSOR_STDOUT="/mnt/d/ZumaTraining/alphazuma-55-headonly-allaim-s99081649-successor.stdout.log"
SUCCESSOR_STDERR="/mnt/d/ZumaTraining/alphazuma-55-headonly-allaim-s99081649-successor.stderr.log"
BOOTSTRAP_ROOT="/mnt/d/ZumaTraining/alphazuma-55-headonly-allaim-s99081647-bootstrap"

expect_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "hash mismatch: $path" >&2
    exit 65
  }
}

expect_sha256 \
  tools/build_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_plan_v1.py \
  fbc9d989103ca80a518301f0c022456997ff5c2060a58843a5f769c74b825e58
expect_sha256 \
  tools/audit_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_prereg_v1.py \
  178a9c4d8570a61fa5a50bf11ee1b1222564c969f5e482608323072c3c751f22
expect_sha256 \
  tools/run_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_v1.py \
  8319c507cd79c5cca930801b5f3e954e3700979464782c31e8186b5d7482a129
expect_sha256 \
  tools/audit_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_v1.py \
  222d4d62376d6325789d1a523b000511f7581d3460b27084ce94b4c262afffac
expect_sha256 \
  tools/build_alphazuma_55_motor_observable_gradual_headonly_allaim_successor_master_v1.py \
  1445963d21cf0c3982544f8b23f68b48548f3d5a49c499e85ba4719b92d150ba
expect_sha256 \
  tools/build_alphazuma_55_formal_seed_registry_v10.py \
  dddd194b0d15777d900a91c97cb06177f279d11d0ea507f46c5e96deadcad10c
expect_sha256 \
  tools/run_alphazuma_55_motor_observable_gradual_headonly_allaim_successor_v1.py \
  54a626c1f67f0ab63cd4587d0567bb570e4a3745f030484d32abb483a835d24a
expect_sha256 \
  tools/audit_alphazuma_55_motor_observable_gradual_headonly_allaim_successor_v1.py \
  5236496ada8beaba8d067dadcedba0ca6b89446ff59e35041c3b1e3f76fbfa72

for target in \
  "$LAUNCH_RECEIPT" "$PLAN" "$PREREG_AUDIT" "$MASTER" "$REGISTRY" \
  "$POST_STDOUT" "$POST_STDERR" "$SUCCESSOR_STDOUT" "$SUCCESSOR_STDERR"; do
  [[ ! -e "$target" ]] || {
    echo "bootstrap target already exists: $target" >&2
    exit 66
  }
done
mkdir -p "$BOOTSTRAP_ROOT"

echo "waiting for factorial round-0 launch artifacts"
trainer_pid=""
for _attempt in $(seq 1 7200); do
  trainer_pid="$(
    pgrep -f "^\.venv/bin/python tools/${TRAINER_NAME} " | head -n 1 || true
  )"
  if [[ -n "$trainer_pid" \
    && -f "$RUN_ROOT/config.json" \
    && -f "$RUN_ROOT/training_status.json" ]]; then
    break
  fi
  sleep 1
done
[[ -n "$trainer_pid" ]] || {
  echo "factorial trainer did not launch within bootstrap window" >&2
  exit 67
}

echo "capturing factorial launch receipt for wsl pid $trainer_pid"
.venv/bin/python \
  tools/capture_alphazuma_55_motor_observable_gradual_headonly_allaim_launch_receipt_v1.py \
  --windows-launcher-pid "$TRAINING_WINDOWS_PID" \
  --wsl-trainer-pid "$trainer_pid" \
  --stdout-log "$TRAINING_STDOUT" \
  --stderr-log "$TRAINING_STDERR" \
  --output "$LAUNCH_RECEIPT" \
  > "$BOOTSTRAP_ROOT/launch-receipt.log"

echo "freezing factorial engineering plan before checkpoint inference"
.venv/bin/python \
  tools/build_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_plan_v1.py \
  --output "$PLAN" \
  > "$BOOTSTRAP_ROOT/plan-builder.log"
plan_sha="sha256:$(sha256sum "$PLAN" | awk '{print $1}')"

.venv/bin/python \
  tools/run_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_v1.py \
  --plan "$PLAN" \
  --expected-plan-sha256 "$plan_sha" \
  --validate-only \
  > "$BOOTSTRAP_ROOT/plan-validation.log"

.venv/bin/python \
  tools/audit_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_prereg_v1.py \
  --plan "$PLAN" \
  --expected-plan-sha256 "$plan_sha" \
  --output "$PREREG_AUDIT" \
  > "$BOOTSTRAP_ROOT/pre-inference-audit.log"

echo "freezing conditional formal master and non-overlap registry"
.venv/bin/python \
  tools/build_alphazuma_55_motor_observable_gradual_headonly_allaim_successor_master_v1.py \
  --expected-engineering-plan-sha256 "$plan_sha" \
  --output "$MASTER" \
  > "$BOOTSTRAP_ROOT/master-builder.log"
master_sha="sha256:$(sha256sum "$MASTER" | awk '{print $1}')"

.venv/bin/python \
  tools/build_alphazuma_55_formal_seed_registry_v10.py \
  --expected-master-sha256 "$master_sha" \
  --output "$REGISTRY" \
  > "$BOOTSTRAP_ROOT/formal-seed-registry.log"

.venv/bin/python \
  tools/run_alphazuma_55_motor_observable_gradual_headonly_allaim_successor_v1.py \
  --master "$MASTER" \
  --expected-master-sha256 "$master_sha" \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
  --validate-only \
  > "$BOOTSTRAP_ROOT/successor-validation.log"

echo "starting conditional successor; formal seeds remain embargoed"
bash tools/launch_alphazuma_55_motor_observable_gradual_headonly_allaim_successor_v1.sh \
  "$master_sha" > "$SUCCESSOR_STDOUT" 2> "$SUCCESSOR_STDERR" &
successor_pid=$!

echo "waiting for remaining frozen training route pid $HEADONLY_TRAINER_PID"
while kill -0 "$HEADONLY_TRAINER_PID" 2>/dev/null; do
  sleep 10
done

echo "starting first-priority factorial engineering evaluation"
bash tools/launch_alphazuma_55_motor_observable_gradual_headonly_allaim_postprocess_v1.sh \
  "$plan_sha" > "$POST_STDOUT" 2> "$POST_STDERR" &
postprocess_pid=$!

echo "postprocess_pid=$postprocess_pid successor_pid=$successor_pid"
set +e
wait "$postprocess_pid"
postprocess_status=$?
wait "$successor_pid"
successor_status=$?
set -e
echo "postprocess_status=$postprocess_status successor_status=$successor_status"
(( postprocess_status == 0 && successor_status == 0 ))
