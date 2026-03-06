#!/bin/bash
#SBATCH --job-name=sgl_trace_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_ROOT="/scratch/qx774/repos/project"
PYTHON_BIN="${PYTHON_BIN:-/scratch/qx774/conda/envs/sgl/bin/python}"
MODEL_PATH="${MODEL_PATH:-NousResearch/Meta-Llama-3-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
CONCURRENCY="${CONCURRENCY:-4}"
WORKLOAD_SPEC="${WORKLOAD_SPEC:-$REPO_ROOT/sglang/python/sglang/srt/experiment_logging/workloads/mixed_small.json}"

export PYTHONPATH="$REPO_ROOT/sglang/python:${PYTHONPATH:-}"

SERVER_LOG="$(mktemp /tmp/sglang_server.XXXXXX.log)"
TRACE_LOG="$(mktemp /tmp/sglang_trace.XXXXXX.log)"
SGLANG_PID=""

cleanup() {
  if [[ -n "${SGLANG_PID}" ]] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
    kill "${SGLANG_PID}" || true
    wait "${SGLANG_PID}" || true
  fi
}
trap cleanup EXIT

echo "Starting SGLang server on ${HOST}:${PORT}"
"${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  >"${SERVER_LOG}" 2>&1 &
SGLANG_PID=$!

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "Server is healthy after ${i}s"
    break
  fi
  sleep 1
  if [[ "${i}" -eq 120 ]]; then
    echo "Server failed health check. Last logs:"
    tail -n 80 "${SERVER_LOG}" || true
    exit 1
  fi
done

echo "Replaying workload: ${WORKLOAD_SPEC}"
"${PYTHON_BIN}" "${REPO_ROOT}/sglang/python/sglang/srt/experiment_logging/run_trace.py" \
  --workload-spec "${WORKLOAD_SPEC}" \
  --endpoint "http://127.0.0.1:${PORT}" \
  --model-path "${MODEL_PATH}" \
  --concurrency "${CONCURRENCY}" \
  --base-dir "${REPO_ROOT}/results" \
  | tee "${TRACE_LOG}"

RUN_DIR="$(awk -F= '/^RUN_DIR=/{print $2}' "${TRACE_LOG}" | tail -n 1)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Could not parse RUN_DIR from run_trace output"
  exit 1
fi

echo "Stopping server"
cleanup
SGLANG_PID=""

echo "Writing summary metrics to ${RUN_DIR}/summary.json"
"${PYTHON_BIN}" "${REPO_ROOT}/sglang/python/sglang/srt/experiment_logging/summarize.py" "${RUN_DIR}"

echo "Completed run: ${RUN_DIR}"
echo "Server log: ${SERVER_LOG}"
