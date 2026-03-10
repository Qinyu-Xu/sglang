#!/usr/bin/env bash
# ==========================================================================
# Generate all workload JSON files for the admission-control experiment.
#
# PURPOSE:
#   Sweep arrival rate × long fraction to find the load regime where
#   the scheduler faces real admission pressure. Run each workload,
#   measure retraction count + throughput + tail latency, and identify
#   the "interesting" regime for the oracle vs. default comparison.
#
# WHAT TO DO WITH THE OUTPUT:
#   1. Start SGLang with your model
#   2. Run each workload JSON through your client driver
#   3. Plot: arrival_rate vs. {retraction_count, throughput, p99_latency}
#      with one line per long_fraction
#   4. The transition point (where retractions start appearing and
#      throughput plateaus) is your target load regime
#
# PARAMETERS TO FILL IN AFTER PROFILING:
#   - ARRIVAL_RATES: update based on load_calculator.py output
#     using your actual KV capacity and service times
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="/scratch/qx774/repos/project/sglang/experiments"
WORKLOADS_DIR="${SCRIPT_DIR}/data/traces"
GENERATOR="${SCRIPT_DIR}/workloads/generate_workload.py"

SEED=42
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TOTAL_REQUESTS=200

# ---------------------------------------------------------------------------
# Arrival rates to sweep (requests/second).
#
# TODO: Replace these with values from load_calculator.py after profiling.
#       These are placeholder estimates assuming ~50K token KV capacity
#       (80GB GPU with --mem-fraction-static 0.10).
#       The range should cover: light → moderate → heavy → overloaded.
# ---------------------------------------------------------------------------
ARRIVAL_RATES=(0.3 0.5 1.0 1.5 2.0 2.5 3.0)

# ---------------------------------------------------------------------------
# Long-request fractions to sweep.
#   0.0 = uniform short (baseline, should show no pathology)
#   0.2 = mild bimodality
#   0.4 = strong bimodality
# ---------------------------------------------------------------------------
LONG_FRACTIONS=(0.0 0.2 0.4)

mkdir -p "${WORKLOADS_DIR}"

echo "=========================================="
echo "Workload generation for load regime sweep"
echo "=========================================="
echo "  Total requests per workload: ${TOTAL_REQUESTS}"
echo "  Arrival rates (req/s):       ${ARRIVAL_RATES[*]}"
echo "  Long fractions:              ${LONG_FRACTIONS[*]}"
echo "  Seed:                        ${SEED}"
echo "  Output dir:                  ${WORKLOADS_DIR}/"
echo ""

COUNT=0

for FRAC in "${LONG_FRACTIONS[@]}"; do
  FRAC_PCT=$(python3 -c "print(int(${FRAC} * 100))")

  for RATE in "${ARRIVAL_RATES[@]}"; do
    # Build a descriptive filename: frac00_rate0.5.json
    RATE_STR=$(echo "${RATE}" | tr '.' 'p')
    OUTFILE="${WORKLOADS_DIR}/frac${FRAC_PCT}_rate${RATE_STR}.json"

    python "${GENERATOR}" \
      --total-requests "${TOTAL_REQUESTS}" \
      --long-fraction "${FRAC}" \
      --arrival-rate "${RATE}" \
      --seed "${SEED}" \
      --model-path "${MODEL_PATH}" \
      --output "${OUTFILE}"

    COUNT=$((COUNT + 1))
  done

  echo ""
done

echo "=========================================="
echo "Generated ${COUNT} workload files in ${WORKLOADS_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Profile: start SGLang, send a few requests, note KV capacity + service times"
echo "  2. Update ARRIVAL_RATES above based on load_calculator.py output"
echo "  3. Re-run this script"
echo "  4. Run each workload through your client driver"
echo "  5. Plot arrival_rate vs. retraction count to find the transition point"
echo "=========================================="