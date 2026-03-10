#!/usr/bin/env bash
# ==========================================================================
# Run the full load-regime sweep.
#
# Iterates through all workload JSON files in data/ and replays each one
# against a running SGLang endpoint. Results go to results/<timestamp>/
#
# PREREQUISITES:
#   1. SGLang server is running:
#        python -m sglang.launch_server \
#          --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#          --port 30000 \
#          --mem-fraction-static 0.10
#
#   2. Workload files have been generated:
#        bash generate_all_workloads.sh
#
# USAGE:
#   bash run_sweep.sh
#   bash run_sweep.sh --endpoint http://localhost:30000
#   bash run_sweep.sh --timeout 900
#   bash run_sweep.sh --data-dir ./my_workloads
#
# OUTPUT:
#   results/
#     sweep_<timestamp>/
#       frac0_rate0p3/        (one dir per workload, from ExperimentLogger)
#       frac0_rate0p5/
#       ...
#       sweep_index.json      (maps workload file → result dir)
# ==========================================================================
set -euo pipefail

EXPERIMENT_DIR="/scratch/qx774/repos/project/sglang/experiments"

# ---------------------------------------------------------------------------
# Defaults (override via CLI flags)
# ---------------------------------------------------------------------------
DATA_DIR="${EXPERIMENT_DIR}/data/traces"
RESULTS_BASE="${EXPERIMENT_DIR}/data/results"
ENDPOINT="http://127.0.0.1:30000"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TIMEOUT=600
COOLDOWN=10          # seconds between runs to let server state settle
REPLAY_SCRIPT="${EXPERIMENT_DIR}/run/run_trace.py"

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)    DATA_DIR="$2";    shift 2 ;;
    --endpoint)    ENDPOINT="$2";    shift 2 ;;
    --model-path)  MODEL_PATH="$2";  shift 2 ;;
    --timeout)     TIMEOUT="$2";     shift 2 ;;
    --cooldown)    COOLDOWN="$2";    shift 2 ;;
    --results-dir) RESULTS_BASE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--data-dir DIR] [--endpoint URL] [--model-path MODEL]"
      echo "          [--timeout SEC] [--cooldown SEC] [--results-dir DIR]"
      exit 0 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: Data directory not found: ${DATA_DIR}"
  echo "Run generate_all_workloads.sh first."
  exit 1
fi

WORKLOAD_FILES=("${DATA_DIR}"/*.json)
if [ ${#WORKLOAD_FILES[@]} -eq 0 ]; then
  echo "ERROR: No .json files found in ${DATA_DIR}/"
  exit 1
fi

# Check server is reachable
echo "Checking endpoint ${ENDPOINT} ..."
if ! curl -sf "${ENDPOINT}/health" > /dev/null 2>&1 && \
   ! curl -sf "${ENDPOINT}/v1/models" > /dev/null 2>&1; then
  echo "WARNING: Could not reach ${ENDPOINT}. Is SGLang running?"
  echo "  Expected: python -m sglang.launch_server --model-path ${MODEL_PATH} --port 30000"
  read -p "Continue anyway? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Create sweep output directory
# ---------------------------------------------------------------------------
SWEEP_TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="${RESULTS_BASE}/sweep_${SWEEP_TS}"
mkdir -p "${SWEEP_DIR}"

echo "=========================================="
echo "Load regime sweep"
echo "=========================================="
echo "  Data dir:     ${DATA_DIR}"
echo "  Results dir:  ${SWEEP_DIR}"
echo "  Endpoint:     ${ENDPOINT}"
echo "  Model:        ${MODEL_PATH}"
echo "  Timeout:      ${TIMEOUT}s"
echo "  Cooldown:     ${COOLDOWN}s between runs"
echo "  Workloads:    ${#WORKLOAD_FILES[@]} files"
echo "=========================================="
echo ""

# Sort workload files so runs happen in a predictable order
# (frac0 before frac20 before frac40, low rate before high rate)
IFS=$'\n' SORTED_FILES=($(sort <<<"${WORKLOAD_FILES[*]}")); unset IFS

# ---------------------------------------------------------------------------
# Sweep index: maps workload file → result directory
# ---------------------------------------------------------------------------
INDEX_FILE="${SWEEP_DIR}/sweep_index.json"
echo "[" > "${INDEX_FILE}"
FIRST_ENTRY=true

TOTAL=${#SORTED_FILES[@]}
CURRENT=0
FAILED=0

for WORKLOAD_FILE in "${SORTED_FILES[@]}"; do
  CURRENT=$((CURRENT + 1))
  BASENAME="$(basename "${WORKLOAD_FILE}" .json)"

  echo "--- [${CURRENT}/${TOTAL}] ${BASENAME} ---"

  # Each workload gets its own subdirectory under the sweep
  RUN_BASE_DIR="${SWEEP_DIR}/${BASENAME}"
  mkdir -p "${RUN_BASE_DIR}"

  # Run the replay
  RUN_OUTPUT=$(python "${REPLAY_SCRIPT}" \
    --workload-spec "${WORKLOAD_FILE}" \
    --endpoint "${ENDPOINT}" \
    --model-path "${MODEL_PATH}" \
    --timeout "${TIMEOUT}" \
    --base-dir "${RUN_BASE_DIR}" \
    2>&1) || {
      echo "  FAILED: ${BASENAME}"
      FAILED=$((FAILED + 1))
      # Still record the failure in the index
      if [ "${FIRST_ENTRY}" = true ]; then
        FIRST_ENTRY=false
      else
        echo "," >> "${INDEX_FILE}"
      fi
      echo "  {\"workload\": \"${BASENAME}\", \"file\": \"${WORKLOAD_FILE}\", \"status\": \"failed\"}" >> "${INDEX_FILE}"
      continue
    }

  # Extract RUN_DIR from output (last line: RUN_DIR=/path/to/run)
  RUN_DIR=$(echo "${RUN_OUTPUT}" | grep "^RUN_DIR=" | tail -1 | cut -d= -f2-)
  echo "  Result: ${RUN_DIR:-unknown}"

  # Append to index
  if [ "${FIRST_ENTRY}" = true ]; then
    FIRST_ENTRY=false
  else
    echo "," >> "${INDEX_FILE}"
  fi
  echo "  {\"workload\": \"${BASENAME}\", \"file\": \"${WORKLOAD_FILE}\", \"run_dir\": \"${RUN_DIR:-unknown}\", \"status\": \"ok\"}" >> "${INDEX_FILE}"

  # Cooldown between runs
  if [ "${CURRENT}" -lt "${TOTAL}" ]; then
    echo "  Cooling down ${COOLDOWN}s ..."
    sleep "${COOLDOWN}"
  fi

  echo ""
done

echo "]" >> "${INDEX_FILE}"

echo "=========================================="
echo "Sweep complete"
echo "  Total:   ${TOTAL}"
echo "  OK:      $((TOTAL - FAILED))"
echo "  Failed:  ${FAILED}"
echo "  Results: ${SWEEP_DIR}/"
echo "  Index:   ${INDEX_FILE}"
echo "=========================================="