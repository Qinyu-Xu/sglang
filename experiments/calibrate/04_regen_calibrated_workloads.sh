#!/usr/bin/env bash
# Step 4 & 5 — Regenerate all workload files with calibrated arrival rates.
#
# Set RATE_* env vars from the output of 03_compute_target_rates.py, then run:
#   bash 04_regen_calibrated_workloads.sh
#
# Example:
#   export RATE_SHORT_ONLY=0.5
#   export RATE_MIXED_20=0.08
#   export RATE_MIXED_40=0.04
#   export RATE_LONG_ONLY=0.02
#   bash 04_regen_calibrated_workloads.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAIRNESS_DIR="$(dirname "${SCRIPT_DIR}")"

PYTHON_BIN="${PYTHON_BIN:-/scratch/qx774/conda/envs/sgl/bin/python}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
SEED="${SEED:-42}"
WORKLOADS_DIR="${WORKLOADS_DIR:-${FAIRNESS_DIR}/workloads}"
GENERATE_PY="${FAIRNESS_DIR}/generate_workload.py"

# Per-workload request counts
N_SHORT_ONLY="${N_SHORT_ONLY:-24}"
N_LONG_ONLY="${N_LONG_ONLY:-12}"
N_MIXED_20="${N_MIXED_20:-20}"
N_MIXED_40="${N_MIXED_40:-20}"

# Rates — override via env vars produced by 03_compute_target_rates.py
RATE_SHORT_ONLY="${RATE_SHORT_ONLY:?Set RATE_SHORT_ONLY from 03_compute_target_rates.py output}"
RATE_LONG_ONLY="${RATE_LONG_ONLY:?Set RATE_LONG_ONLY from 03_compute_target_rates.py output}"
RATE_MIXED_20="${RATE_MIXED_20:?Set RATE_MIXED_20 from 03_compute_target_rates.py output}"
RATE_MIXED_40="${RATE_MIXED_40:?Set RATE_MIXED_40 from 03_compute_target_rates.py output}"

mkdir -p "${WORKLOADS_DIR}"

echo "Regenerating workloads with calibrated arrival rates"
echo "  workloads dir : ${WORKLOADS_DIR}"
echo "  seed          : ${SEED}"
echo "---------------------------------------------------"

"${PYTHON_BIN}" "${GENERATE_PY}" \
    --total-requests "${N_SHORT_ONLY}" --long-fraction 0.0 \
    --arrival-rate "${RATE_SHORT_ONLY}" --seed "${SEED}" \
    --model-path "${MODEL_PATH}" --output "${WORKLOADS_DIR}/short_only.json"

"${PYTHON_BIN}" "${GENERATE_PY}" \
    --total-requests "${N_LONG_ONLY}" --long-fraction 1.0 \
    --arrival-rate "${RATE_LONG_ONLY}" --seed "${SEED}" \
    --model-path "${MODEL_PATH}" --output "${WORKLOADS_DIR}/long_only.json"

"${PYTHON_BIN}" "${GENERATE_PY}" \
    --total-requests "${N_MIXED_20}" --long-fraction 0.2 \
    --arrival-rate "${RATE_MIXED_20}" --seed "${SEED}" \
    --model-path "${MODEL_PATH}" --output "${WORKLOADS_DIR}/mixed_20.json"

"${PYTHON_BIN}" "${GENERATE_PY}" \
    --total-requests "${N_MIXED_40}" --long-fraction 0.4 \
    --arrival-rate "${RATE_MIXED_40}" --seed "${SEED}" \
    --model-path "${MODEL_PATH}" --output "${WORKLOADS_DIR}/mixed_40.json"

echo "---------------------------------------------------"
echo "Done. Workloads written to ${WORKLOADS_DIR}/"
echo ""
echo "Next: run the fairness experiment"
echo "  bash ../run_fairness_experiment.sh"
