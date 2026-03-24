#!/usr/bin/env bash
# ==========================================================================
# Full experiment: start server → run utilization sweep → stop server.
#
# USAGE:
#   bash run_experiment.sh
#   bash run_experiment.sh --utilizations "0.8 1.0 1.2" --long-fraction 0.4
#   bash run_experiment.sh --clip-max-new-tokens-chat 128
#
# Server parameters are forwarded to start_server.sh.
# Sweep parameters are forwarded to run_utilization_sweep.sh.
# Shared parameters (--model-path, --port/--endpoint,
# --schedule-conservativeness, --clip-max-tokens-estimation) are passed
# to both scripts so they stay consistent.
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

START_SERVER_SCRIPT="${SCRIPT_DIR}/start_server.sh"
SWEEP_SCRIPT="${SCRIPT_DIR}/run_utilization_sweep.sh"

# ---------------------------------------------------------------------------
# Shared defaults (kept in sync with both sub-scripts)
# ---------------------------------------------------------------------------
MODEL_PATH="Qwen/Qwen3-8B"
PORT="${PORT:-$((30000 + UID % 10000))}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------------------------------------------------------
# Server-only defaults
# ---------------------------------------------------------------------------
MEM_FRACTION=0.5
DTYPE="auto"
ENABLE_RADIX_CACHE=false
ENABLE_PRIORITY_SCHEDULING=false
SKIP_LONG_ON_NO_TOKEN=false
MAX_CONCURRENT_CHAT=0
SCHEDULE_CONSERVATIVENESS=1.0
INIT_NEW_TOKEN_RATIO=""
MIN_NEW_TOKEN_RATIO_FACTOR=""
NEW_TOKEN_RATIO_DECAY_STEPS=""
CLIP_MAX_TOKENS_ESTIMATION=""   # maps to SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION
SERVER_READY_TIMEOUT=300
LOG_FILE=""                     # auto-set by start_server.sh if empty

# ---------------------------------------------------------------------------
# Sweep-only defaults
# ---------------------------------------------------------------------------
UTILIZATIONS="1.0"
LONG_FRACTION=0.2
TOTAL_REQUESTS=500
SEED=42
REQUEST_TIMEOUT=900
COOLDOWN=15
RESULTS_DIR="${EXPERIMENT_DIR}/data/results"
TRACE_DIR="${EXPERIMENT_DIR}/data/trace"
CHAT_PRIORITY=0
CLIP_MAX_NEW_TOKENS=4096
CLIP_MAX_NEW_TOKENS_CHAT=""

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    # shared
    --model-path)                 MODEL_PATH="$2";                shift 2 ;;
    --port)                       PORT="$2";                      shift 2 ;;
    --schedule-conservativeness)  SCHEDULE_CONSERVATIVENESS="$2"; shift 2 ;;
    --python)                     PYTHON_BIN="$2";                shift 2 ;;

    # server
    --mem-fraction)               MEM_FRACTION="$2";              shift 2 ;;
    --dtype)                      DTYPE="$2";                     shift 2 ;;
    --enable-radix-cache)         ENABLE_RADIX_CACHE=true;        shift ;;
    --enable-priority-scheduling) ENABLE_PRIORITY_SCHEDULING=true; shift ;;
    --skip-long-on-no-token)      SKIP_LONG_ON_NO_TOKEN=true;      shift ;;
    --max-concurrent-chat)        MAX_CONCURRENT_CHAT="$2";        shift 2 ;;
    --init-token-ratio)           INIT_NEW_TOKEN_RATIO="$2";      shift 2 ;;
    --min-token-ratio-factor)     MIN_NEW_TOKEN_RATIO_FACTOR="$2"; shift 2 ;;
    --token-ratio-decay-steps)    NEW_TOKEN_RATIO_DECAY_STEPS="$2"; shift 2 ;;
    --clip-max-tokens-estimation) CLIP_MAX_TOKENS_ESTIMATION="$2"; shift 2 ;;
    --server-ready-timeout)       SERVER_READY_TIMEOUT="$2";      shift 2 ;;
    --log-file)                   LOG_FILE="$2";                  shift 2 ;;

    # sweep
    --utilizations)               UTILIZATIONS="$2";              shift 2 ;;
    --long-fraction)              LONG_FRACTION="$2";             shift 2 ;;
    --total-requests)             TOTAL_REQUESTS="$2";            shift 2 ;;
    --seed)                       SEED="$2";                      shift 2 ;;
    --request-timeout)            REQUEST_TIMEOUT="$2";           shift 2 ;;
    --cooldown)                   COOLDOWN="$2";                  shift 2 ;;
    --results-dir)                RESULTS_DIR="$2";               shift 2 ;;
    --trace-dir)                  TRACE_DIR="$2";                 shift 2 ;;
    --chat-priority)              CHAT_PRIORITY="$2";             shift 2 ;;
    --clip-max-new-tokens)        CLIP_MAX_NEW_TOKENS="$2";       shift 2 ;;
    --clip-max-new-tokens-chat)   CLIP_MAX_NEW_TOKENS_CHAT="$2";  shift 2 ;;

    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Shared:"
      echo "  --model-path                  STR    HuggingFace model path (default: Qwen/Qwen3-8B)"
      echo "  --port                        INT    Server port (default: 30000 + UID%%10000)"
      echo "  --schedule-conservativeness   FLOAT  Scale init_new_token_ratio (default: 1.0)"
      echo "  --python                      PATH   Python binary (default: python)"
      echo ""
      echo "Server (start_server.sh):"
      echo "  --mem-fraction                FLOAT  GPU memory fraction for KV cache (default: 0.5)"
      echo "  --dtype                       STR    Model dtype (default: auto)"
      echo "  --enable-radix-cache                 Enable prefix/radix cache (default: disabled)"
      echo "  --enable-priority-scheduling         Enable priority scheduling (default: disabled)"
      echo "  --init-token-ratio            FLOAT  SGLANG_INIT_NEW_TOKEN_RATIO (default: 0.7)"
      echo "  --min-token-ratio-factor      FLOAT  SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR (default: 0.14)"
      echo "  --token-ratio-decay-steps     INT    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS (default: 600)"
      echo "  --clip-max-tokens-estimation  INT    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION (default: 4096)"
      echo "  --server-ready-timeout        INT    Seconds to wait for server ready (default: 300)"
      echo "  --log-file                    PATH   Server log file (default: /tmp/sglang_<port>.log)"
      echo ""
      echo "Sweep (run_utilization_sweep.sh):"
      echo "  --utilizations                FLOAT... Space-separated targets (default: '1.0')"
      echo "  --long-fraction               FLOAT  Fraction of long requests (default: 0.2)"
      echo "  --total-requests              INT    Requests per workload (default: 500)"
      echo "  --seed                        INT    RNG seed (default: 42)"
      echo "  --request-timeout             INT    Per-request timeout seconds (default: 900)"
      echo "  --cooldown                    INT    Seconds between runs (default: 15)"
      echo "  --results-dir                 DIR    Base results directory"
      echo "  --trace-dir                   DIR    Directory for generated trace files"
      echo "  --chat-priority               INT    Chat request priority (0 = disabled)"
      echo "  --clip-max-new-tokens         INT    CLIP_MAX_NEW_TOKENS for index (default: 4096)"
      echo "  --clip-max-new-tokens-chat    INT    CLIP_MAX_NEW_TOKENS_CHAT for index (default: same as above)"
      exit 0 ;;
    *) echo "ERROR: Unknown flag: $1"; exit 1 ;;
  esac
done

ENDPOINT="http://127.0.0.1:${PORT}"
PID_FILE="/tmp/sglang_${PORT}.pid"
[[ -z "${LOG_FILE}" ]] && LOG_FILE="/tmp/sglang_${PORT}.log"

# ---------------------------------------------------------------------------
# Stop server on exit (success or failure)
# ---------------------------------------------------------------------------
stop_server() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid=$(cat "${PID_FILE}")
    echo ""
    echo "Stopping server (PID ${pid})..."
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    rm -f "${PID_FILE}"
  fi
}
trap stop_server EXIT

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------
SERVER_ARGS=(
  --model-path "${MODEL_PATH}"
  --port "${PORT}"
  --mem-fraction "${MEM_FRACTION}"
  --dtype "${DTYPE}"
  --schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}"
  --timeout "${SERVER_READY_TIMEOUT}"
  --log-file "${LOG_FILE}"
  --python "${PYTHON_BIN}"
)
[[ "${ENABLE_RADIX_CACHE}" == true ]]         && SERVER_ARGS+=(--enable-radix-cache)
[[ "${ENABLE_PRIORITY_SCHEDULING}" == true ]] && SERVER_ARGS+=(--enable-priority-scheduling)
[[ "${SKIP_LONG_ON_NO_TOKEN}" == true ]]      && SERVER_ARGS+=(--skip-long-on-no-token)
[[ "${MAX_CONCURRENT_CHAT}" -gt 0 ]] 2>/dev/null && SERVER_ARGS+=(--max-concurrent-chat "${MAX_CONCURRENT_CHAT}")
[[ -n "${INIT_NEW_TOKEN_RATIO}" ]]            && SERVER_ARGS+=(--init-token-ratio "${INIT_NEW_TOKEN_RATIO}")
[[ -n "${MIN_NEW_TOKEN_RATIO_FACTOR}" ]]      && SERVER_ARGS+=(--min-token-ratio-factor "${MIN_NEW_TOKEN_RATIO_FACTOR}")
[[ -n "${NEW_TOKEN_RATIO_DECAY_STEPS}" ]]     && SERVER_ARGS+=(--token-ratio-decay-steps "${NEW_TOKEN_RATIO_DECAY_STEPS}")
[[ -n "${CLIP_MAX_TOKENS_ESTIMATION}" ]]      && SERVER_ARGS+=(--clip-max-tokens-estimation "${CLIP_MAX_TOKENS_ESTIMATION}")

bash "${START_SERVER_SCRIPT}" "${SERVER_ARGS[@]}"

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
SWEEP_ARGS=(
  --model-path "${MODEL_PATH}"
  --endpoint "${ENDPOINT}"
  --utilizations "${UTILIZATIONS}"
  --long-fraction "${LONG_FRACTION}"
  --total-requests "${TOTAL_REQUESTS}"
  --seed "${SEED}"
  --timeout "${REQUEST_TIMEOUT}"
  --cooldown "${COOLDOWN}"
  --results-dir "${RESULTS_DIR}"
  --trace-dir "${TRACE_DIR}"
  --schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}"
  --chat-priority "${CHAT_PRIORITY}"
  --clip-max-new-tokens "${CLIP_MAX_NEW_TOKENS}"
  --server-log "${LOG_FILE}"
)
[[ -n "${CLIP_MAX_NEW_TOKENS_CHAT}" ]] && SWEEP_ARGS+=(--clip-max-new-tokens-chat "${CLIP_MAX_NEW_TOKENS_CHAT}")

# Signal any background test.py processes that the experiment is starting
pgrep -u "$(whoami)" -f "test\.py" | while read -r pid; do
  kill -USR2 "${pid}" 2>/dev/null && echo "Sent SIGUSR2 to PID ${pid} (test.py)"
done

bash "${SWEEP_SCRIPT}" "${SWEEP_ARGS[@]}"

# Signal that the experiment has completed
pgrep -u "$(whoami)" -f "test\.py" | while read -r pid; do
  kill -USR1 "${pid}" 2>/dev/null && echo "Sent SIGUSR1 to PID ${pid} (test.py)"
done
