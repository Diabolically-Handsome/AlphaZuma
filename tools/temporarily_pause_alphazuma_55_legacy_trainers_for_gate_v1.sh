#!/usr/bin/env bash
set -euo pipefail

declare -A expected_groups=(
  [3786]=3786
  [3797]=3797
  [12280]=12280
  [31773]=8488
  [39873]=39873
  [41683]=41683
  [330473]=330473
)

groups=()
for pid in 3786 3797 12280 31773 39873 41683 330473; do
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  observed_group="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
  if [[ "${cmdline}" != *"train_alphazuma_55.py"* ]]; then
    echo "refusing to pause unexpected pid ${pid}: ${cmdline}" >&2
    exit 1
  fi
  if [[ "${observed_group}" != "${expected_groups[${pid}]}" ]]; then
    echo "refusing to pause pid ${pid} with unexpected pgid ${observed_group}" >&2
    exit 1
  fi
  groups+=("-${observed_group}")
done

resume_log=/mnt/d/ZumaTraining/alphazuma-55-legacy-trainers-temporary-gate-pause-s99081654-auto-resume.log
nohup bash -c '
  sleep 300
  kill -CONT -- -3786 -3797 -12280 -8488 -39873 -41683 -330473 2>/dev/null || true
  date -u +"auto_resumed_utc=%Y-%m-%dT%H:%M:%SZ"
' >"${resume_log}" 2>&1 </dev/null &
resume_pid=$!

kill -STOP -- "${groups[@]}"
echo "pause_status=STOPPED"
echo "auto_resume_pid=${resume_pid}"
echo "auto_resume_after_seconds=300"
printf 'paused_process_groups=%s\n' "${groups[*]}"
