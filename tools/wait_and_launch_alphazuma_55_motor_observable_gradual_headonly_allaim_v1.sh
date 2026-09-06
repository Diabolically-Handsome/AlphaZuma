#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Documents/祖玛"

BLOCKING_PID=390568
RUN_DIR="/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-v1"
RECEIPT="diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-launch-receipt-v1.json"

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
  diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-preregistration-v1.json \
  bcede95c539b97ea9a440ccee5f68e8715ce6b3dd3c9ad3eac2b7651d28c3470
expect_sha256 \
  diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-independent-prereg-audit-v1.json \
  b0c17f5a4d96bf940fbe267b0a1472bcc31b6282119d02f551744a36f18a430c
expect_sha256 \
  tools/distill_alphazuma_55_motor_observable_gradual_headonly_allaim_v1.py \
  a1087452f49bb1fdc516394e1e9f07a6e8955490ddfd5025798dbdd2710ec372
expect_sha256 \
  tools/build_alphazuma_55_motor_observable_gradual_headonly_allaim_v1.py \
  33547d3149a64aec6972e28910455a3a388cb410f05d3522510b771a9f554a4a

echo "waiting for frozen cuda:1 predecessor pid $BLOCKING_PID"
while kill -0 "$BLOCKING_PID" 2>/dev/null; do
  sleep 10
done

[[ ! -e "$RUN_DIR" ]] || {
  echo "factorial run directory already exists" >&2
  exit 66
}
[[ ! -e "$RECEIPT" ]] || {
  echo "factorial launch receipt already exists" >&2
  exit 67
}

ready=0
for _attempt in $(seq 1 60); do
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_total_kib="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
  swap_free_kib="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
  root_available="$(df -B1 --output=avail / | tail -n 1 | tr -d ' ')"
  mapfile -t limits < <(
    nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits \
      | awk '{printf "%.0f\n", $1}'
  )
  mapfile -t temperatures < <(
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits \
      | awk '{printf "%d\n", $1}'
  )
  swap_used_kib=$((swap_total_kib - swap_free_kib))
  if (( available_kib >= 12 * 1024 * 1024 )) \
    && (( swap_used_kib == 0 )) \
    && (( root_available >= 12 * 1024 * 1024 * 1024 )) \
    && [[ "${limits[*]}" == "550 250" ]] \
    && (( temperatures[0] < 85 )) \
    && (( temperatures[1] < 85 )); then
    ready=1
    break
  fi
  sleep 10
done

[[ "$ready" == 1 ]] || {
  echo "resources did not return to the frozen launch envelope" >&2
  exit 68
}

.venv/bin/python \
  tools/distill_alphazuma_55_motor_observable_gradual_headonly_allaim_v1.py \
  --preregistration \
  diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-preregistration-v1.json \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge" \
  --validate-only

echo "launching frozen factorial route"
exec bash tools/launch_alphazuma_55_motor_observable_gradual_headonly_allaim_v1.sh
