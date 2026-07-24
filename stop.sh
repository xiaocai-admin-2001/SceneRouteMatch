#!/usr/bin/env bash
set -euo pipefail
PID="/home/cpk/road_segment_test_service/service.pid"
if [[ ! -f "$PID" ]]; then
  echo "not running"
  exit 0
fi
pid="$(cat "$PID")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "stopped pid=$pid"
else
  echo "stale pid=$pid"
fi
rm -f "$PID"
