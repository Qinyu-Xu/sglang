#!/usr/bin/env bash
# Step 2 & 3 — Sweep arrival rates for short_only and long_only.
#
# Generates tiny workloads at each rate, replays them, and prints
# a latency table so you can pick a target rate before the full experiment.
#
# Usage (server must be running):
#   bash 02_sweep_arrival_rate.sh
#
# Override via env vars:
#   RATES_SHORT="0.5 1.0 2.0 4.0"  (req/s)
#   RATES_LONG="0.05 0.1 0.2 0.5"
#   SWEEP_N=30          total requests per sweep run
#   SEED=42
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAIRNESS_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "$(dirname "${FAIRNESS_DIR}")")"

PYTHON_BIN="${PYTHON_BIN:-/scratch/qx774/conda/envs/sgl/bin/python}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:30000}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-${REPO_ROOT}/sglang/results}"
CONCURRENCY="${CONCURRENCY:-4}"

RATES_SHORT="${RATES_SHORT:-0.25 0.5 1.0 2.0 4.0}"
RATES_LONG="${RATES_LONG:-0.05 0.1 0.2 0.5 1.0}"
SWEEP_N="${SWEEP_N:-30}"
SEED="${SEED:-42}"

GENERATE_PY="${FAIRNESS_DIR}/generate_workload.py"
RUN_TRACE_PY="${FAIRNESS_DIR}/run_trace.py"

SWEEP_DIR="$(mktemp -d /tmp/sglang_sweep.XXXXXX)"
trap 'rm -rf "${SWEEP_DIR}"' EXIT

export PYTHONPATH="${REPO_ROOT}/sglang/python:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Helper: run one workload and extract p50/p95 completion latency per class
# ---------------------------------------------------------------------------
run_and_summarize() {
    local workload_file="$1"
    local label="$2"

    local run_output
    run_output="$(
        "${PYTHON_BIN}" "${RUN_TRACE_PY}" \
            --workload-spec "${workload_file}" \
            --endpoint      "${ENDPOINT}" \
            --model-path    "${MODEL_PATH}" \
            --concurrency   "${CONCURRENCY}" \
            --base-dir      "${BASE_RESULTS_DIR}" \
            2>/dev/null
    )"

    local run_dir
    run_dir="$(echo "${run_output}" | awk -F= '/^RUN_DIR=/{print $2}' | tail -n1)"

    # Extract p50/p95 completion latency from requests.jsonl using inline Python
    "${PYTHON_BIN}" - "${run_dir}" "${label}" <<'PYEOF'
import json, sys, statistics
from pathlib import Path

run_dir = Path(sys.argv[1])
target_label = sys.argv[2]  # "short_only" or "long_only" — used in display only

records = [json.loads(l) for l in (run_dir / "requests.jsonl").read_text().splitlines() if l.strip()]
completions = sorted(
    r["finish_ts"] - r["submit_ts"]
    for r in records
    if r.get("status") == "success" and r.get("finish_ts") and r.get("submit_ts")
)
errors = sum(1 for r in records if r.get("status") != "success")

def pct(vals, p):
    if not vals: return float("nan")
    idx = max(0, int(len(vals) * p / 100) - 1)
    return vals[idx]

p50 = pct(completions, 50)
p95 = pct(completions, 95)
print(f"    p50={p50:6.2f}s  p95={p95:6.2f}s  errors={errors}/{len(records)}")
PYEOF
}

# ---------------------------------------------------------------------------
# Sweep short_only
# ---------------------------------------------------------------------------
echo "========================================"
echo "SWEEP: short_only"
echo "  rates (req/s): ${RATES_SHORT}"
echo "  n per run:     ${SWEEP_N}"
echo "========================================"
printf "  %-12s  %s\n" "arrival_rate" "completion latency"
printf "  %-12s  %s\n" "------------" "------------------"

for rate in ${RATES_SHORT}; do
    wl="${SWEEP_DIR}/short_${rate}.json"
    "${PYTHON_BIN}" "${GENERATE_PY}" \
        --total-requests "${SWEEP_N}" \
        --long-fraction  0.0 \
        --arrival-rate   "${rate}" \
        --seed           "${SEED}" \
        --model-path     "${MODEL_PATH}" \
        --output         "${wl}" \
        >/dev/null

    printf "  %-12s" "${rate}"
    run_and_summarize "${wl}" "short_only"
done

echo ""

# ---------------------------------------------------------------------------
# Sweep long_only
# ---------------------------------------------------------------------------
echo "========================================"
echo "SWEEP: long_only"
echo "  rates (req/s): ${RATES_LONG}"
echo "  n per run:     ${SWEEP_N}"
echo "========================================"
printf "  %-12s  %s\n" "arrival_rate" "completion latency"
printf "  %-12s  %s\n" "------------" "------------------"

for rate in ${RATES_LONG}; do
    wl="${SWEEP_DIR}/long_${rate}.json"
    "${PYTHON_BIN}" "${GENERATE_PY}" \
        --total-requests "${SWEEP_N}" \
        --long-fraction  1.0 \
        --arrival-rate   "${rate}" \
        --seed           "${SEED}" \
        --model-path     "${MODEL_PATH}" \
        --output         "${wl}" \
        >/dev/null

    printf "  %-12s" "${rate}"
    run_and_summarize "${wl}" "long_only"
done

echo ""
echo "Done. Pick rates where p50 is stable and p95/p50 ratio is < 2."
echo "Then run: python 03_compute_target_rates.py --short-svc-time S --long-svc-time L"
