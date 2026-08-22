#!/usr/bin/env bash
set -e

# Run against the already-running stack: `docker compose up --build -d`
HOST="http://localhost:8000"   # Nginx, not a gateway replica directly
DURATION=30s
USERS=500
SPAWN_RATE=50
OUT_DIR="benchmarks/results"
mkdir -p "$OUT_DIR"

echo "=== Load-testing $HOST for ${DURATION} with ${USERS} users ==="
locust -f benchmarks/locustfile.py --host "$HOST" \
  --headless -u "$USERS" -r "$SPAWN_RATE" --run-time "$DURATION" \
  --csv="$OUT_DIR/gateway_run"

echo "Done. CSVs are in $OUT_DIR/"
echo
echo "For a genuine 1M-request-class run, use Locust's distributed mode:"
echo "  locust -f benchmarks/locustfile.py --master --host $HOST"
echo "  locust -f benchmarks/locustfile.py --worker --master-host <master-ip>   # x N machines"
