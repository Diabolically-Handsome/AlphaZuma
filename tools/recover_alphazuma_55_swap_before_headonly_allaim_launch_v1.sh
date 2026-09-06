#!/usr/bin/env bash
set -euo pipefail

BLOCKING_PID=390568
SWAP_DEVICE=/dev/sdc
MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))

echo "waiting for cuda:1 predecessor pid $BLOCKING_PID before swap recovery"
while kill -0 "$BLOCKING_PID" 2>/dev/null; do
  sleep 5
done
sleep 2

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
swap_total_kib="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
swap_free_kib="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
swap_used_kib=$((swap_total_kib - swap_free_kib))
echo "before available_kib=$available_kib swap_used_kib=$swap_used_kib"

if (( swap_used_kib == 0 )); then
  echo "swap already clear; no mutation required"
  exit 0
fi
if (( available_kib < MIN_AVAILABLE_KIB )); then
  echo "insufficient available memory for bounded swap reload" >&2
  exit 69
fi
if [[ ! -b "$SWAP_DEVICE" ]]; then
  echo "expected WSL swap device is absent: $SWAP_DEVICE" >&2
  exit 70
fi

swapoff "$SWAP_DEVICE"
trap 'swapon "$SWAP_DEVICE" 2>/dev/null || true' EXIT
swapon "$SWAP_DEVICE"
trap - EXIT

available_after_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
swap_total_after_kib="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
swap_free_after_kib="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
swap_used_after_kib=$((swap_total_after_kib - swap_free_after_kib))
echo "after available_kib=$available_after_kib swap_used_kib=$swap_used_after_kib"
(( available_after_kib >= 12 * 1024 * 1024 ))
(( swap_used_after_kib == 0 ))
