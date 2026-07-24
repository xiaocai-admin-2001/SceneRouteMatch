#!/usr/bin/env bash
set -euo pipefail
cd /home/cpk/road_segment_test_service
PORT="${EVENT_TEST_PORT:-19000}"
BACKEND="${ROAD_SEGMENT_URL:-http://127.0.0.1:8990/api/extract_road}"
THREADS="${EVENT_TEST_THREADS:-4}"
BIN="/home/cpk/road_segment_test_service/build/road_segment_test_service"
LOG="/home/cpk/road_segment_test_service/service.log"
PID="/home/cpk/road_segment_test_service/service.pid"

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "already running pid=$(cat "$PID")"
  exit 0
fi

nohup env EVENT_TEST_PORT="$PORT" ROAD_SEGMENT_URL="$BACKEND" EVENT_TEST_THREADS="$THREADS" "$BIN" >"$LOG" 2>&1 &
echo $! > "$PID"
echo "started pid=$(cat "$PID") port=$PORT"
