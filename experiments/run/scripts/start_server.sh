#!/usr/bin/env bash
# ==========================================================================
# Start the SGLang server and wait until it is ready to serve requests.
#
# Usage:
#   bash start_server.sh
#   bash start_server.sh --port 39942
#   bash start_server.sh --model-path /path/to/model --mem-fraction 0.85
#   bash start_server.sh --enable-radix-cache   # turn on prefix caching
#   bash start_server.sh --dry-run              # print command without running
#
# The script:
#   1. Checks the port is free (errors if already in use)
#   2. Launches SGLang in a new process group (setsid)
#   3. Polls /health_generate until the model is fully loaded
#   4. Prints the endpoint URL on success
#
# To stop the server later:
#   kill -- -$(cat /tmp/sglang_<port>.pid)   # kill process group
#   or:
#   pkill -9 -f sglang.launch_server
# ==========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH="Qwen/Qwen3-8B"
PORT="${PORT:-$((30000 + UID % 10000))}"
MEM_FRACTION=0.5
DTYPE="auto"
ENABLE_RADIX_CACHE=false
INSTANT_ACCEPT_CHAT=false

# Future token estimation controls
# --schedule-conservativeness: scales init_new_token_ratio (1.0 = default, >1 = more conservative)
# SGLANG_INIT_NEW_TOKEN_RATIO:          initial fraction of max_new_tokens assumed to be used (default 0.7)
# SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR:    floor for the adaptive ratio (default 0.14)
# SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS:   steps to decay from init to min (default 600)
# SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION: hard cap on estimated remaining tokens (default 4096)
SCHEDULE_CONSERVATIVENESS=1.0
INIT_NEW_TOKEN_RATIO=""          # empty = use SGLang default (0.7)
MIN_NEW_TOKEN_RATIO_FACTOR=""    # empty = use SGLang default (0.14)
NEW_TOKEN_RATIO_DECAY_STEPS=""   # empty = use SGLang default (600)
CLIP_MAX_NEW_TOKENS_ESTIMATION="" # empty = use SGLang default (4096)

READY_TIMEOUT=300        # seconds to wait for /health_generate
LOG_FILE=""              # auto-set below if empty
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN=false

EXPERIMENT_DIR="/scratch/qx774/repos/project/sglang/experiments"

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)        MODEL_PATH="$2";        shift 2 ;;
    --port)              PORT="$2";              shift 2 ;;
    --mem-fraction)      MEM_FRACTION="$2";      shift 2 ;;
    --dtype)             DTYPE="$2";             shift 2 ;;
    --enable-radix-cache)           ENABLE_RADIX_CACHE=true;             shift ;;
    --schedule-conservativeness)    SCHEDULE_CONSERVATIVENESS="$2";      shift 2 ;;
    --instant-accept-chat)          INSTANT_ACCEPT_CHAT=true;            shift 1 ;;
    --init-token-ratio)             INIT_NEW_TOKEN_RATIO="$2";           shift 2 ;;
    --min-token-ratio-factor)       MIN_NEW_TOKEN_RATIO_FACTOR="$2";     shift 2 ;;
    --token-ratio-decay-steps)      NEW_TOKEN_RATIO_DECAY_STEPS="$2";    shift 2 ;;
    --clip-max-tokens-estimation)   CLIP_MAX_NEW_TOKENS_ESTIMATION="$2"; shift 2 ;;
    --timeout)                      READY_TIMEOUT="$2";                  shift 2 ;;
    --log-file)          LOG_FILE="$2";          shift 2 ;;
    --python)            PYTHON_BIN="$2";        shift 2 ;;
    --dry-run)           DRY_RUN=true;           shift ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --model-path        STR   HuggingFace model path (default: Qwen/Qwen3-8B)"
      echo "  --port              INT   Server port (default: 30000 + UID%%10000 = $((30000 + UID % 10000)))"
      echo "  --mem-fraction      FLOAT GPU memory fraction for KV cache (default: 0.5)"
      echo "  --dtype             STR   Model dtype (default: auto)"
      echo "  --enable-radix-cache               Enable prefix/radix cache (default: disabled)"
      echo "  --schedule-conservativeness FLOAT  Scale init_new_token_ratio (default: 1.0)"
      echo "  --init-token-ratio          FLOAT  SGLANG_INIT_NEW_TOKEN_RATIO (default: 0.7)"
      echo "  --min-token-ratio-factor    FLOAT  SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR (default: 0.14)"
      echo "  --token-ratio-decay-steps   INT    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS (default: 600)"
      echo "  --clip-max-tokens-estimation INT   SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION (default: 4096)"
      echo "  --timeout                   INT    Seconds to wait for server ready (default: 300)"
      echo "  --log-file          PATH  Server log file (default: /tmp/sglang_<port>.log)"
      echo "  --python            PATH  Python binary (default: python)"
      echo "  --dry-run                 Print launch command without executing"
      exit 0 ;;
    *) echo "ERROR: Unknown flag: $1"; exit 1 ;;
  esac
done

ENDPOINT="http://127.0.0.1:${PORT}"
PID_FILE="/tmp/sglang_${PORT}.pid"
[[ -z "${LOG_FILE}" ]] && LOG_FILE="/tmp/sglang_${PORT}.log"

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" == true ]]; then
  echo "Would run:"
  echo "  setsid ${PYTHON_BIN} -m sglang.launch_server \\"
  echo "    --model-path ${MODEL_PATH} \\"
  echo "    --port ${PORT} \\"
  echo "    --mem-fraction-static ${MEM_FRACTION} \\"
  echo "    --dtype ${DTYPE}"
  [[ "${ENABLE_RADIX_CACHE}" == false ]] && echo "    --disable-radix-cache"
  echo "  Log: ${LOG_FILE}"
  echo "  Endpoint: ${ENDPOINT}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Check port is free
# ---------------------------------------------------------------------------
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "ERROR: Port ${PORT} is already in use."
  echo "  To find what's using it: ss -tlnp | grep :${PORT}"
  echo "  To kill it:              pkill -9 -f sglang.launch_server"
  exit 1
fi

# ---------------------------------------------------------------------------
# Check if server is already running at this endpoint
# ---------------------------------------------------------------------------
if curl -sf "${ENDPOINT}/health_generate" > /dev/null 2>&1; then
  echo "Server already running at ${ENDPOINT}"
  echo "  ENDPOINT=${ENDPOINT}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Launch server in new process group
# ---------------------------------------------------------------------------
echo "Starting SGLang server..."
echo "  Model:         ${MODEL_PATH}"
echo "  Port:          ${PORT}"
echo "  Mem fraction:  ${MEM_FRACTION}"
echo "  Radix cache:              ${ENABLE_RADIX_CACHE}"
echo "  Schedule conservativeness: ${SCHEDULE_CONSERVATIVENESS}"
[[ -n "${INIT_NEW_TOKEN_RATIO}" ]]          && echo "  Init token ratio:          ${INIT_NEW_TOKEN_RATIO}"
[[ -n "${MIN_NEW_TOKEN_RATIO_FACTOR}" ]]    && echo "  Min token ratio factor:    ${MIN_NEW_TOKEN_RATIO_FACTOR}"
[[ -n "${NEW_TOKEN_RATIO_DECAY_STEPS}" ]]   && echo "  Token ratio decay steps:   ${NEW_TOKEN_RATIO_DECAY_STEPS}"
[[ -n "${CLIP_MAX_NEW_TOKENS_ESTIMATION}" ]] && echo "  Clip max tokens est.:      ${CLIP_MAX_NEW_TOKENS_ESTIMATION}"
echo "  Log:                       ${LOG_FILE}"
echo ""

# Set env vars for future token estimation if overrides were provided
[[ -n "${INIT_NEW_TOKEN_RATIO}" ]]           && export SGLANG_INIT_NEW_TOKEN_RATIO="${INIT_NEW_TOKEN_RATIO}"
[[ -n "${MIN_NEW_TOKEN_RATIO_FACTOR}" ]]     && export SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR="${MIN_NEW_TOKEN_RATIO_FACTOR}"
[[ -n "${NEW_TOKEN_RATIO_DECAY_STEPS}" ]]    && export SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS="${NEW_TOKEN_RATIO_DECAY_STEPS}"
[[ -n "${CLIP_MAX_NEW_TOKENS_ESTIMATION}" ]] && export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION="${CLIP_MAX_NEW_TOKENS_ESTIMATION}"

EXTRA_ARGS=()
[[ "${ENABLE_RADIX_CACHE}" == false ]] && EXTRA_ARGS+=(--disable-radix-cache)
EXTRA_ARGS+=(--schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}")
[[ "${INSTANT_ACCEPT_CHAT}" == true ]] && EXTRA_ARGS+=(--instant-accept-chat)
EXTRA_ARGS+=(--enable-metrics)

setsid "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --dtype "${DTYPE}" \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

SGLANG_PID=$!
echo "${SGLANG_PID}" > "${PID_FILE}"
echo "  PID: ${SGLANG_PID}  (saved to ${PID_FILE})"
echo ""

# ---------------------------------------------------------------------------
# Cleanup trap — kill process group on Ctrl-C during wait
# ---------------------------------------------------------------------------
cleanup() {
  echo ""
  echo "Interrupted — stopping server (PID ${SGLANG_PID})..."
  kill -- "-${SGLANG_PID}" 2>/dev/null || kill "${SGLANG_PID}" 2>/dev/null || true
  exit 1
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# Wait for /health_generate
# ---------------------------------------------------------------------------
echo "Waiting for server to be ready (timeout: ${READY_TIMEOUT}s)..."
for i in $(seq 1 "${READY_TIMEOUT}"); do
  # Fast-fail if process died
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo ""
    echo "ERROR: Server process died (PID ${SGLANG_PID})"
    echo "Last 40 lines of ${LOG_FILE}:"
    tail -n 40 "${LOG_FILE}" || true
    exit 1
  fi

  if curl -sf "${ENDPOINT}/health_generate" > /dev/null 2>&1; then
    echo ""
    echo "Server is ready."
    echo "  ENDPOINT=${ENDPOINT}"
    echo "  PID_FILE=${PID_FILE}"
    echo "  LOG_FILE=${LOG_FILE}"
    echo ""
    echo "To stop:  kill -- -\$(cat ${PID_FILE})"
    trap - INT TERM
    exit 0
  fi

  # Progress indicator every 10s
  if (( i % 10 == 0 )); then
    echo "  ${i}s elapsed — still waiting..."
  fi
  sleep 1
done

echo ""
echo "ERROR: Server did not become ready within ${READY_TIMEOUT}s."
echo "Last 40 lines of ${LOG_FILE}:"
tail -n 40 "${LOG_FILE}" || true
kill -- "-${SGLANG_PID}" 2>/dev/null || kill "${SGLANG_PID}" 2>/dev/null || true
exit 1
