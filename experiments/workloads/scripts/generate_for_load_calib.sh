#!/usr/bin/env bash
# ==========================================================================
# Generate profiling traces for measuring isolated service times.
#
# These runs send requests one at a time (very low arrival rate) so each
# request runs alone with zero contention. The measured completion latency
# = true service time for that request class.
#
# Usage:
#   bash generate_profiling_traces.sh
#   bash generate_profiling_traces.sh --num-requests 30
#
# Outputs:
#   data/profile/short_profile.json   — N isolated short requests
#   data/profile/long_profile.json    — N isolated long requests
# ==========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NUM_REQUESTS=20
SEED=42
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ENDPOINT="http://127.0.0.1:30000"

SCRIPT_DIR="/scratch/qx774/repos/project/sglang/experiments"
OUTPUT_DIR="${SCRIPT_DIR}/data/trace"
GENERATOR="${SCRIPT_DIR}/workloads/generate_workload.py"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --seed)         SEED="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Arrival rate low enough that requests never overlap.
# 0.01 req/s = one request every 100 seconds.
# Even the longest request (~300s worst case) will finish before the next arrives
# because we're profiling, not benchmarking.
ARRIVAL_RATE_LONG=0.01
ARRIVAL_RATE_SHORT=0.1

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Generating profiling traces"
echo "  Requests per class: ${NUM_REQUESTS}"
echo "  Arrival rate:       ${ARRIVAL_RATE_LONG} req/s (isolated)"

echo "  Output dir:         ${OUTPUT_DIR}"
echo "============================================"
echo ""

# --- Short-only profiling trace ---
echo "--- Short requests (${NUM_REQUESTS}) ---"
python "${GENERATOR}" \
  --total-requests "${NUM_REQUESTS}" \
  --long-fraction 0.0 \
  --arrival-rate "${ARRIVAL_RATE_SHORT}" \
  --arrival-mode uniform \
  --seed "${SEED}" \
  --model-path "${MODEL_PATH}" \
  --endpoint "${ENDPOINT}" \
  --output "${OUTPUT_DIR}/short_profile.json"

echo ""

# --- Long-only profiling trace ---
echo "--- Long requests (${NUM_REQUESTS}) ---"
python "${GENERATOR}" \
  --total-requests "${NUM_REQUESTS}" \
  --long-fraction 1.0 \
  --arrival-rate "${ARRIVAL_RATE_LONG}" \
  --arrival-mode uniform \
  --seed "${SEED}" \
  --model-path "${MODEL_PATH}" \
  --endpoint "${ENDPOINT}" \
  --output "${OUTPUT_DIR}/long_profile.json"

echo ""
echo "============================================"
echo "Done. Run these traces against SGLang:"
echo ""
echo "  python run/run_trace.py --workload-spec ${OUTPUT_DIR}/short_profile.json"
echo "  python run/run_trace.py --workload-spec ${OUTPUT_DIR}/long_profile.json"
echo ""
echo "Then compute S_short, S_long, KV_short, KV_long from the results"
echo "and update generate_for_utilization.sh with the new constants."
echo "============================================"