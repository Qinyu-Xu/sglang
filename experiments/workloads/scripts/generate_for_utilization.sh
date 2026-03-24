#!/usr/bin/env bash
# ==========================================================================
# Generate a workload trace for a target KV-cache utilization level.
#
# Usage:
#   bash generate_for_utilization.sh 0.8
#   bash generate_for_utilization.sh 1.0 --long-fraction 0.4
#   bash generate_for_utilization.sh 1.2 --total-requests 300
#
# The first positional arg is the target utilization u.
# Lambda is computed as:
#
#   λ = (u × C) / [(1-f) × S_short × KV_short  +  f × S_long × KV_long]
#
# where C = KV cache capacity (tokens), S = mean service time (seconds),
# KV = mean KV tokens held per request (prompt + output).
# ==========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Hardcoded from profiling (Qwen/Qwen3-8B on A100-80GB)
#
# Profile runs:
#   short: run_20260310_091638  (5 isolated short reqs, 128 output tokens each)
#   long:  run_20260310_092307  (5 isolated long reqs, mean ~1965 output tokens)
# ---------------------------------------------------------------------------
S_SHORT=0.7        # mean service time for short requests (seconds)
S_LONG=93        # mean service time for long requests (seconds)
KV_SHORT=43        # mean KV tokens per short request (prompt + output)
KV_LONG=4246        # mean KV tokens per long request (prompt + output)
C=45622             # KV cache capacity in tokens (back-calculated from profile)

# ---------------------------------------------------------------------------
# Defaults (overridable via flags)
# ---------------------------------------------------------------------------
LONG_FRACTION=0.2
TOTAL_REQUESTS=200
SEED=42
MODEL_PATH="Qwen/Qwen3-8B"
ENDPOINT="http://127.0.0.1:30000"

SCRIPT_DIR="/scratch/qx774/repos/project/sglang/experiments"
OUTPUT_DIR="${SCRIPT_DIR}/data/trace"
GENERATOR="${SCRIPT_DIR}/workloads/generate_workload.py"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <utilization> [--long-fraction F] [--total-requests N] [--seed S]"
  echo ""
  echo "Examples:"
  echo "  $0 0.8                        # u=0.8, f=0.2 (default)"
  echo "  $0 1.0 --long-fraction 0.4    # u=1.0, f=0.4"
  echo "  $0 1.2 --total-requests 300   # u=1.2, 300 requests"
  exit 1
fi

UTILIZATION="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --long-fraction)  LONG_FRACTION="$2"; shift 2 ;;
    --total-requests) TOTAL_REQUESTS="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Compute λ using python3 (floating point math)
# ---------------------------------------------------------------------------
LAMBDA=$(python3 -c "
u = ${UTILIZATION}
C = ${C}
f = ${LONG_FRACTION}
S_short = ${S_SHORT}
S_long = ${S_LONG}
KV_short = ${KV_SHORT}
KV_long = ${KV_LONG}

denom = (1 - f) * S_short * KV_short + f * S_long * KV_long
lam = (u * C) / denom
print(f'{lam:.4f}')
")

# Build output filename: u0p8_f20.json
U_STR=$(echo "${UTILIZATION}" | tr '.' 'p')
F_PCT=$(python3 -c "print(int(${LONG_FRACTION} * 100))")
OUTFILE="${OUTPUT_DIR}/u${U_STR}_f${F_PCT}.json"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Target utilization:  u = ${UTILIZATION}"
echo "Long fraction:       f = ${LONG_FRACTION}"
echo "Computed λ:          ${LAMBDA} req/s"
echo "Total requests:      ${TOTAL_REQUESTS}"
echo "Output:              ${OUTFILE}"
echo "--------------------------------------------"
echo "Constants:"
echo "  S_short  = ${S_SHORT}s    KV_short = ${KV_SHORT} tokens"
echo "  S_long   = ${S_LONG}s   KV_long  = ${KV_LONG} tokens"
echo "  C        = ${C} tokens"
echo "============================================"
echo ""

python "${GENERATOR}" \
  --total-requests "${TOTAL_REQUESTS}" \
  --long-fraction "${LONG_FRACTION}" \
  --arrival-rate "${LAMBDA}" \
  --seed "${SEED}" \
  --model-path "${MODEL_PATH}" \
  --endpoint "${ENDPOINT}" \
  --output "${OUTFILE}"

echo ""
echo "Done: ${OUTFILE}"
