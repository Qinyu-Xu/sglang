#!/usr/bin/env bash
set -euo pipefail

: <<'EXPERIMENT_NOTES'
Fixed:
1. Framework
2. Model
3. # GPU
4. Concurrency
5. Server Flags
6. Prompt Templates
7. Total Request Count

Vary:
1. Long-request Fraction
2. Maybe load regime later

Measure:
1. TTFT p50/p95 by class
2. Completion latency p50/p95 by class
3. Slowdown p50 by class
4. Success/Error/Timeout Counts

Log for Debugging:
1. Request Order
2. Prompt Len
3. Output Len
4. Target Output Len
5. Max New Tokens Per Request
6. Prompt Type
EXPERIMENT_NOTES

# [FIXED] framework: SGLang (python -m sglang.launch_server)
# [FIXED] 1 GPU: pin to a single visible device
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_ROOT="${REPO_ROOT:-/scratch/qx774/repos/project}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/qx774/conda/envs/sgl/bin/python}"
MODEL_PATH="${MODEL_PATH:-NousResearch/Meta-Llama-3-8B-Instruct}" # [FIXED] model
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:${PORT}}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-${REPO_ROOT}/sglang/results}"
WORKLOAD_DIR="${WORKLOAD_DIR:-${REPO_ROOT}/sglang/experiments/serving_fairness/workloads}"
CONCURRENCY="${CONCURRENCY:-4}" # [FIXED] concurrency

# [FIXED] server flags for single-node, non-distributed serving
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
EXTRA_SERVER_FLAGS="${EXTRA_SERVER_FLAGS:-}"
read -r -a EXTRA_SERVER_FLAGS_ARR <<<"${EXTRA_SERVER_FLAGS}"

SCRIPT_DIR="${REPO_ROOT}/sglang/experiments/serving_fairness"
RUN_TRACE_PY="${SCRIPT_DIR}/run_trace.py"
SUMMARIZE_PY="${REPO_ROOT}/sglang/experiments/serving_fairness/summarize.py"
COMPARE_PY="${SCRIPT_DIR}/compare_runs.py"

export PYTHONPATH="${REPO_ROOT}/sglang/python:${PYTHONPATH:-}"

SERVER_LOG="$(mktemp /tmp/sglang_fairness_server.XXXXXX.log)"
TRACE_LOG="$(mktemp /tmp/sglang_fairness_trace.XXXXXX.log)"
RUN_DIRS=()
SGLANG_PID=""

WORKLOADS=(
  "short_only.json"
  "long_only.json"
  "mixed_20.json"
  "mixed_40.json"
)
# [FIXED] prompt templates + total request count are defined in workload specs
# [VARIED] class composition / long-request fraction via workload file choice

cleanup() {
  if [[ -n "${SGLANG_PID}" ]] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
    kill "${SGLANG_PID}" || true
    wait "${SGLANG_PID}" || true
  fi
}
trap cleanup EXIT

echo "Starting SGLang server: ${MODEL_PATH} at ${HOST}:${PORT}"
"${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --dp-size "${DP_SIZE}" \
  "${EXTRA_SERVER_FLAGS_ARR[@]}" \
  >"${SERVER_LOG}" 2>&1 &
SGLANG_PID=$!

echo "Waiting for health endpoint..."
for i in $(seq 1 180); do
  if curl -sf "${ENDPOINT}/health" >/dev/null; then
    echo "Server is healthy after ${i}s"
    break
  fi
  sleep 1
  if [[ "${i}" -eq 180 ]]; then
    echo "Server failed health check. Last server logs:"
    tail -n 120 "${SERVER_LOG}" || true
    exit 1
  fi
done

for workload_file in "${WORKLOADS[@]}"; do
  workload_path="${WORKLOAD_DIR}/${workload_file}"
  if [[ ! -f "${workload_path}" ]]; then
    echo "Missing workload spec: ${workload_path}"
    exit 1
  fi

  echo "Replaying workload: ${workload_file}"
  run_output="$(
    "${PYTHON_BIN}" "${RUN_TRACE_PY}" \
      --workload-spec "${workload_path}" \
      --endpoint "${ENDPOINT}" \
      --model-path "${MODEL_PATH}" \
      --concurrency "${CONCURRENCY}" \
      --base-dir "${BASE_RESULTS_DIR}" \
      | tee -a "${TRACE_LOG}"
  )"

  run_dir="$(echo "${run_output}" | awk -F= '/^RUN_DIR=/{print $2}' | tail -n 1)"
  if [[ -z "${run_dir}" ]]; then
    echo "Could not parse RUN_DIR for workload ${workload_file}"
    exit 1
  fi
  RUN_DIRS+=("${run_dir}")

  "${PYTHON_BIN}" "${SUMMARIZE_PY}" "${run_dir}"
  # [OBSERVED] per-run metrics in summary.json:
  # - TTFT p50/p95/p99 by class
  # - completion latency p50/p95/p99 by class
  # - success/error/timeout counts
  # [DEBUG] per-request fields in requests.jsonl:
  # - request_order, prompt_len, output_len,
  # - target_output_len, max_new_tokens_per_request, prompt_type
  echo "Completed workload ${workload_file}: ${run_dir}"
done

timestamp="$(date +%Y%m%d_%H%M%S)"
comparison_csv="${BASE_RESULTS_DIR}/fairness_comparison_${timestamp}.csv"
comparison_md="${BASE_RESULTS_DIR}/fairness_comparison_${timestamp}.md"

echo "Comparing runs..."
"${PYTHON_BIN}" "${COMPARE_PY}" \
  "${RUN_DIRS[@]}" \
  --output-csv "${comparison_csv}" \
  --output-md "${comparison_md}"
# [OBSERVED] slowdown_vs_baseline (p50 completion) by class in comparison outputs

echo ""
echo "Fairness experiment complete."
echo "Run directories:"
for run_dir in "${RUN_DIRS[@]}"; do
  echo "  - ${run_dir}"
done
echo "Comparison CSV: ${comparison_csv}"
echo "Comparison MD:  ${comparison_md}"
echo "Server log:     ${SERVER_LOG}"
