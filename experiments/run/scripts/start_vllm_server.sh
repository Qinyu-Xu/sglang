#!/usr/bin/env bash
# ==========================================================================
# Start the vLLM server and wait until it is ready to serve requests.
#
# Usage:
#   bash start_vllm_server.sh
#   bash start_vllm_server.sh --port 39942
#   bash start_vllm_server.sh --model-path /path/to/model --gpu-memory-utilization 0.85
#   bash start_vllm_server.sh --enable-prefix-caching   # turn on prefix caching
#   bash start_vllm_server.sh --dry-run                 # print command without running
#
# The script:
#   1. Checks the port is free (errors if already in use)
#   2. Launches vLLM in a new process group (setsid)
#   3. Polls /v1/models until the model is fully loaded
#   4. Prints the endpoint URL on success
#
# To stop the server later:
#   kill -- -$(cat /tmp/vllm_<port>.pid)   # kill process group
#   or:
#   pkill -9 -f vllm.entrypoints.openai.api_server
# ==========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH="Qwen/Qwen3-8B"
PORT="${PORT:-$((30000 + UID % 10000))}"
GPU_MEMORY_UTILIZATION=0.5
DTYPE="auto"
ENABLE_PREFIX_CACHING=false
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

READY_TIMEOUT=300        # seconds to wait for /v1/models
LOG_FILE=""              # auto-set below if empty
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN=false

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)              MODEL_PATH="$2";              shift 2 ;;
    --port)                    PORT="$2";                    shift 2 ;;
    --gpu-memory-utilization)  GPU_MEMORY_UTILIZATION="$2";  shift 2 ;;
    --dtype)                   DTYPE="$2";                   shift 2 ;;
    --cuda-devices)            CUDA_DEVICES="$2";            shift 2 ;;
    --enable-prefix-caching)   ENABLE_PREFIX_CACHING=true;   shift ;;
    --timeout)                 READY_TIMEOUT="$2";           shift 2 ;;
    --log-file)                LOG_FILE="$2";                shift 2 ;;
    --python)                  PYTHON_BIN="$2";              shift 2 ;;
    --dry-run)                 DRY_RUN=true;                 shift ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --model-path              STR   HuggingFace model path (default: Qwen/Qwen3-8B)"
      echo "  --port                    INT   Server port (default: 30000 + UID%%10000 = $((30000 + UID % 10000)))"
      echo "  --gpu-memory-utilization  FLOAT GPU memory fraction for KV cache (default: 0.9)"
      echo "  --dtype                   STR   Model dtype (default: auto)"
      echo "  --cuda-devices            STR   CUDA_VISIBLE_DEVICES value (default: 0)"
      echo "  --enable-prefix-caching         Enable prefix caching (default: disabled)"
      echo "  --timeout                 INT   Seconds to wait for server ready (default: 300)"
      echo "  --log-file                PATH  Server log file (default: /tmp/vllm_<port>.log)"
      echo "  --python                  PATH  Python binary (default: python)"
      echo "  --dry-run                       Print launch command without executing"
      exit 0 ;;
    *) echo "ERROR: Unknown flag: $1"; exit 1 ;;
  esac
done

ENDPOINT="http://127.0.0.1:${PORT}"
PID_FILE="/tmp/vllm_${PORT}.pid"
[[ -z "${LOG_FILE}" ]] && LOG_FILE="/tmp/vllm_${PORT}.log"

EXTRA_ARGS=()
[[ "${ENABLE_PREFIX_CACHING}" == false ]] && EXTRA_ARGS+=(--no-enable-prefix-caching)

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" == true ]]; then
  echo "Would run:"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} setsid ${PYTHON_BIN} -m vllm.entrypoints.openai.api_server \\"
  echo "    --model ${MODEL_PATH} \\"
  echo "    --port ${PORT} \\"
  echo "    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \\"
  echo "    --dtype ${DTYPE} \\"
  [[ "${ENABLE_PREFIX_CACHING}" == false ]] && echo "    --no-enable-prefix-caching \\"
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
  echo "  To kill it:              pkill -9 -f vllm.entrypoints.openai.api_server"
  exit 1
fi

# ---------------------------------------------------------------------------
# Check if server is already running at this endpoint
# ---------------------------------------------------------------------------
if curl -sf "${ENDPOINT}/v1/models" > /dev/null 2>&1; then
  echo "Server already running at ${ENDPOINT}"
  echo "  ENDPOINT=${ENDPOINT}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Launch server in new process group
# ---------------------------------------------------------------------------
echo "Starting vLLM server..."
echo "  Model:                   ${MODEL_PATH}"
echo "  Port:                    ${PORT}"
echo "  GPU memory utilization:  ${GPU_MEMORY_UTILIZATION}"
echo "  CUDA_VISIBLE_DEVICES:    ${CUDA_DEVICES}"
echo "  Prefix caching:          ${ENABLE_PREFIX_CACHING}"
echo "  Log:                     ${LOG_FILE}"
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
setsid "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype "${DTYPE}" \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

VLLM_PID=$!
echo "${VLLM_PID}" > "${PID_FILE}"
echo "  PID: ${VLLM_PID}  (saved to ${PID_FILE})"
echo ""

# ---------------------------------------------------------------------------
# Cleanup trap — kill process group on Ctrl-C during wait
# ---------------------------------------------------------------------------
cleanup() {
  echo ""
  echo "Interrupted — stopping server (PID ${VLLM_PID})..."
  kill -- "-${VLLM_PID}" 2>/dev/null || kill "${VLLM_PID}" 2>/dev/null || true
  exit 1
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# Wait for /v1/models
# ---------------------------------------------------------------------------
echo "Waiting for server to be ready (timeout: ${READY_TIMEOUT}s)..."
for i in $(seq 1 "${READY_TIMEOUT}"); do
  # Fast-fail if process died
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo ""
    echo "ERROR: Server process died (PID ${VLLM_PID})"
    echo "Last 40 lines of ${LOG_FILE}:"
    tail -n 40 "${LOG_FILE}" || true
    exit 1
  fi

  if curl -sf "${ENDPOINT}/v1/models" > /dev/null 2>&1; then
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
kill -- "-${VLLM_PID}" 2>/dev/null || kill "${VLLM_PID}" 2>/dev/null || true
exit 1
