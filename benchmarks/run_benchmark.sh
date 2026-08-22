set -e

HOST="http://localhost:8000"
DURATION=30s
USERS=100
SPAWN_RATE=20
OUT_DIR="benchmarks/results"
mkdir -p "$OUT_DIR"

run_config () {
  local name=$1
  local max_batch=$2
  echo "=== Running config: $name (MAX_BATCH_SIZE=$max_batch) ==="

  MAX_BATCH_SIZE=$max_batch uvicorn app.main:app --host 0.0.0.0 --port 8000 &
  SERVER_PID=$!
  sleep 5   # wait for model to load

  locust -f benchmarks/locustfile.py --host "$HOST" \
    --headless -u $USERS -r $SPAWN_RATE --run-time $DURATION \
    --csv="$OUT_DIR/$name"

  kill $SERVER_PID
  wait $SERVER_PID 2>/dev/null || true
  sleep 2
}

run_config "baseline_batch1"   1
run_config "dynamic_batch8"    8
run_config "dynamic_batch16"   16
run_config "dynamic_batch32"   32

echo "All runs complete. CSVs are in $OUT_DIR/"