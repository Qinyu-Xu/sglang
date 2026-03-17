#!/usr/bin/env bash
# ==========================================================================
# Generate workloads at target utilization levels and run the sweep.
#
# For each utilization u, calls generate_for_utilization.sh to compute
# lambda and generate a trace, then replays it against a running server.
#
# PREREQUISITES:
#   SGLang server must already be running. This script does NOT start it.
#   Check with:
#     curl http://127.0.0.1:30000/health_generate
#
# USAGE:
#   bash run_utilization_sweep.sh
#   bash run_utilization_sweep.sh --utilizations "0.6 0.8 1.0 1.2"
#   bash run_utilization_sweep.sh --long-fraction 0.4 --total-requests 300
#   bash run_utilization_sweep.sh --endpoint http://localhost:39942
#
# OUTPUT:
#   data/trace/u{U}_f{F}.json     — generated workload per utilization
#   data/results/sweep_<ts>/      — one run_* subdir per utilization
# ==========================================================================
set -euo pipefail

EXPERIMENT_DIR="/scratch/qx774/repos/project/sglang/experiments"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
UTILIZATIONS="0.8 1.0 1.2"
LONG_FRACTION=0.2
TOTAL_REQUESTS=500
SEED=42
ENDPOINT="http://127.0.0.1:39942"
MODEL_PATH="Qwen/Qwen3-8B"
TIMEOUT=900
COOLDOWN=15
SCHEDULE_CONSERVATIVENESS=""   # empty = not recorded (read from running server)
SERVER_LOG=""                 # if set, server log slice is copied per run

GENERATE_SCRIPT="${EXPERIMENT_DIR}/workloads/scripts/generate_for_utilization.sh"
REPLAY_SCRIPT="${EXPERIMENT_DIR}/run/run_trace.py"
TRACE_DIR="${EXPERIMENT_DIR}/data/trace"
RESULTS_BASE="${EXPERIMENT_DIR}/data/results"

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --utilizations)    UTILIZATIONS="$2";    shift 2 ;;
    --long-fraction)   LONG_FRACTION="$2";   shift 2 ;;
    --total-requests)  TOTAL_REQUESTS="$2";  shift 2 ;;
    --seed)            SEED="$2";            shift 2 ;;
    --endpoint)        ENDPOINT="$2";        shift 2 ;;
    --model-path)      MODEL_PATH="$2";      shift 2 ;;
    --timeout)         TIMEOUT="$2";         shift 2 ;;
    --cooldown)        COOLDOWN="$2";        shift 2 ;;
    --results-dir)                RESULTS_BASE="$2";             shift 2 ;;
    --trace-dir)                  TRACE_DIR="$2";                shift 2 ;;
    --schedule-conservativeness)  SCHEDULE_CONSERVATIVENESS="$2"; shift 2 ;;
    --server-log)                 SERVER_LOG="$2";                shift 2 ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --utilizations   FLOAT...  Space-separated utilization targets (default: '0.8 1.0 1.2')"
      echo "  --long-fraction  FLOAT     Fraction of long requests (default: 0.2)"
      echo "  --total-requests INT       Requests per workload (default: 200)"
      echo "  --seed           INT       RNG seed (default: 42)"
      echo "  --endpoint       URL       SGLang endpoint (default: http://127.0.0.1:30000)"
      echo "  --model-path     STR       Model path (default: Qwen/Qwen3-8B)"
      echo "  --timeout        INT       Per-request timeout seconds (default: 900)"
      echo "  --cooldown       INT       Seconds between runs (default: 15)"
      echo "  --results-dir                DIR    Base results directory"
      echo "  --trace-dir                  DIR    Directory for generated trace files"
      echo "  --schedule-conservativeness  FLOAT  Recorded in sweep index for reference (set at server start)"
      exit 0 ;;
    *) echo "ERROR: Unknown flag: $1"; exit 1 ;;
  esac
done

read -ra U_LIST <<< "${UTILIZATIONS}"

# ---------------------------------------------------------------------------
# Server health check — fail fast, do not start server
# ---------------------------------------------------------------------------
echo "Checking server at ${ENDPOINT} ..."
if curl -sf "${ENDPOINT}/health_generate" > /dev/null 2>&1; then
  echo "  Server ready (health_generate OK)"
elif curl -sf "${ENDPOINT}/v1/models" > /dev/null 2>&1; then
  echo "  Server reachable (v1/models OK)"
else
  echo "ERROR: Server not reachable at ${ENDPOINT}"
  echo "  Start SGLang first:"
  echo "    python -m sglang.launch_server \\"
  echo "      --model-path ${MODEL_PATH} \\"
  echo "      --port 30000"
  exit 1
fi

mkdir -p "${TRACE_DIR}" "${RESULTS_BASE}"

# One sweep directory for this entire run
SWEEP_TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="${RESULTS_BASE}/sweep_${SWEEP_TS}"
mkdir -p "${SWEEP_DIR}"

F_PCT=$(python3 -c "print(int(${LONG_FRACTION} * 100))")

echo ""
echo "=========================================="
echo "Utilization sweep"
echo "=========================================="
echo "  Utilizations:               ${UTILIZATIONS}"
echo "  Long fraction:              ${LONG_FRACTION} (${F_PCT}%)"
echo "  Total requests:             ${TOTAL_REQUESTS}"
echo "  Endpoint:                   ${ENDPOINT}"
echo "  Model:                      ${MODEL_PATH}"
echo "  Timeout:                    ${TIMEOUT}s / request"
echo "  Cooldown:                   ${COOLDOWN}s between runs"
echo "  Schedule conservativeness:  ${SCHEDULE_CONSERVATIVENESS:-"(server default)"}"
echo "  Trace dir:                  ${TRACE_DIR}"
echo "  Results dir:                ${SWEEP_DIR}"
echo "=========================================="
echo ""

TOTAL=${#U_LIST[@]}
CURRENT=0
FAILED=0

# Sweep index JSON
INDEX_FILE="${SWEEP_DIR}/sweep_index.json"
echo "[" > "${INDEX_FILE}"
FIRST_ENTRY=true

for U in "${U_LIST[@]}"; do
  CURRENT=$((CURRENT + 1))
  U_STR=$(echo "${U}" | tr '.' 'p')
  TRACE_FILE="${TRACE_DIR}/u${U_STR}_f${F_PCT}.json"
  RUN_BASE_DIR="${SWEEP_DIR}/u${U_STR}_f${F_PCT}"
  mkdir -p "${RUN_BASE_DIR}"

  echo "--- [${CURRENT}/${TOTAL}] u=${U} ---"

  # --- Generate workload ---
  echo "  Generating trace (u=${U}, f=${LONG_FRACTION}) ..."
  bash "${GENERATE_SCRIPT}" "${U}" \
    --long-fraction "${LONG_FRACTION}" \
    --total-requests "${TOTAL_REQUESTS}" \
    --seed "${SEED}" \
    --output-dir "${TRACE_DIR}" \
    || {
      echo "  ERROR: workload generation failed for u=${U}"
      FAILED=$((FAILED + 1))
      continue
    }

  if [[ ! -f "${TRACE_FILE}" ]]; then
    echo "  ERROR: expected trace file not found: ${TRACE_FILE}"
    FAILED=$((FAILED + 1))
    continue
  fi

  # --- Replay trace ---
  echo "  Running trace ..."
  REPLAY_EXTRA_ARGS=()
  [[ -n "${SERVER_LOG}" ]] && REPLAY_EXTRA_ARGS+=(--server-log "${SERVER_LOG}")

  RUN_OUTPUT=$(python "${REPLAY_SCRIPT}" \
    --workload-spec "${TRACE_FILE}" \
    --endpoint "${ENDPOINT}" \
    --model-path "${MODEL_PATH}" \
    --timeout "${TIMEOUT}" \
    --base-dir "${RUN_BASE_DIR}" \
    "${REPLAY_EXTRA_ARGS[@]}" \
    2>&1) || {
      echo "  ERROR: replay failed for u=${U}"
      echo "${RUN_OUTPUT}" | tail -5
      FAILED=$((FAILED + 1))
      _append_index() {
        [[ "${FIRST_ENTRY}" == true ]] && FIRST_ENTRY=false || echo "," >> "${INDEX_FILE}"
        echo "  {\"u\": ${U}, \"trace\": \"${TRACE_FILE}\", \"status\": \"failed\"}" >> "${INDEX_FILE}"
      }
      _append_index
      continue
    }

  RUN_DIR=$(echo "${RUN_OUTPUT}" | grep "^RUN_DIR=" | tail -1 | cut -d= -f2-)
  echo "  Result: ${RUN_DIR:-${RUN_BASE_DIR}}"

  [[ "${FIRST_ENTRY}" == true ]] && FIRST_ENTRY=false || echo "," >> "${INDEX_FILE}"
  CONSERV_JSON="${SCHEDULE_CONSERVATIVENESS:-null}"
  echo "  {\"u\": ${U}, \"trace\": \"${TRACE_FILE}\", \"run_dir\": \"${RUN_DIR:-unknown}\", \"schedule_conservativeness\": ${CONSERV_JSON}, \"status\": \"ok\"}" >> "${INDEX_FILE}"

  # Cooldown (skip after last run)
  if [[ "${CURRENT}" -lt "${TOTAL}" ]]; then
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
