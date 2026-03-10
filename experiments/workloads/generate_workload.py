#!/usr/bin/env python3
"""
Generate workload JSON files for SGLang admission-control experiments.

==========================================================================
INPUT PARAMETERS
==========================================================================

Workload composition:
  --total-requests  N     Total number of requests to generate (default: 200)
  --long-fraction   F     Fraction of requests that are long/reasoning [0.0–1.0]

Arrival timing:
  --arrival-rate    λ     Overall mean arrival rate in requests/second.
                          This is THE key variable to sweep across experiments.
                          Requests arrive as a Poisson process with rate λ.
                          Compute λ from load_calculator.py using profiled
                          service times and KV cache capacity.

Reproducibility:
  --seed            S     Random seed (default: 42)

Metadata:
  --model-path      M     HuggingFace model path (for output metadata)
  --endpoint        E     Server endpoint URL (for output metadata)
  --output          O     Output JSON file path

==========================================================================
OUTPUT FORMAT
==========================================================================

The output JSON contains a flat list of requests, each with:
  - request_id:          Unique ID (req_0000, req_0001, ...)
  - request_order:       Submission order (0-indexed)
  - class_label:         "short" or "long"
  - prompt_text:         The actual prompt string
  - submit_time_offset:  Seconds after experiment start to submit this request
  - max_new_tokens:      Hard cap on output length (128 for short, 8192 for long)
  - prompt_type:         "chat" or "reasoning"

The client driver should:
  1. Read the request list
  2. Send each request at its submit_time_offset (relative to experiment start)
  3. NOT enforce any concurrency cap — let the server scheduler decide
  4. Record per-request metrics (TTFT, completion latency, output length)

==========================================================================
WORKLOAD DESIGN RATIONALE
==========================================================================

Short requests:  Simple factual questions → ~40-80 output tokens
Long requests:   Reasoning/math problems → ~512-4096 output tokens (lognormal)

The bimodality comes from mixing these two classes. The arrival rate
controls how much memory pressure the scheduler faces. Use the load
calculator to find the arrival rate that targets 60-80% KV utilization.

To sweep load factor, generate multiple workloads with different
--arrival-rate values while keeping everything else fixed.

==========================================================================
"""

import argparse
import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# Topic pools
# ---------------------------------------------------------------------------

SHORT_TOPICS = [
    "What is TTFT?",
    "What is the capital of France?",
    "How does TCP/IP work?",
    "What is a floating point number?",
    "Explain what a hash table is.",
    "What is the difference between RAM and ROM?",
    "What does HTTP stand for?",
    "What is a binary search?",
    "What is the speed of light?",
    "Who invented the telephone?",
    "What is a compiler?",
    "What is machine learning?",
    "What is an API?",
    "What is the boiling point of water?",
    "What is a linked list?",
    "What is Git?",
    "What does CPU stand for?",
    "What is recursion?",
    "What is a database index?",
    "What is latency in networking?",
]

REASONING_TOPICS = [
    "Prove that the square root of 2 is irrational.",
    "Solve: A train leaves city A at 60 mph and another leaves city B at 90 mph toward each other 300 miles apart. When do they meet?",
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes.",
    "Explain the time complexity of merge sort and derive it from the recurrence relation.",
    "How would you design a distributed key-value store that handles node failures?",
    "Prove by induction that the sum of the first N natural numbers is N*(N+1)/2.",
    "Given a binary tree, write an algorithm to check if it is height-balanced.",
    "Analyze the Knight's Tour problem and describe an efficient algorithm to solve it.",
    "Explain how the RSA encryption algorithm works and why it is secure.",
    "Design an LRU cache with O(1) get and put operations.",
    "Solve the Tower of Hanoi problem for N disks and analyze its time complexity.",
    "Explain how gradient descent works in neural network training.",
    "Write and analyze an algorithm to find the longest common subsequence of two strings.",
    "How would you implement a thread-safe singleton pattern in Python?",
    "Derive the formula for compound interest and explain its exponential growth.",
    "Explain the CAP theorem and its implications for distributed systems.",
    "Write a regex engine that supports `.` and `*` operators.",
    "Analyze why quicksort has O(n log n) average case but O(n^2) worst case.",
    "Design a URL shortener service that can handle 100 million URLs.",
    "Explain how a B-tree works and why databases use it for indexing.",
]

PROMPT_TEMPLATES = {
    "short": "Give a concise answer in 1-2 sentences: {topic}",
    "long": "Reason step by step and provide a detailed analysis: {topic}",
}


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_poisson_offsets(n: int, arrival_rate: float, rng: np.random.Generator) -> list[float]:
    """
    Generate arrival times from a Poisson process.

    Inter-arrival times ~ Exponential(1/λ) where λ = arrival_rate.
    First request arrives at t=0.
    """
    if n <= 0:
        return []
    inter_arrivals = rng.exponential(scale=1.0 / arrival_rate, size=n)
    offsets = np.cumsum(inter_arrivals)
    # Shift so first request is at t=0
    offsets = offsets - offsets[0]
    return [round(float(t), 3) for t in offsets]


def uniform_offsets(n: int, arrival_rate: float) -> list[float]:
    """
    Generate evenly spaced arrival times.

    Inter-arrival = 1/λ exactly. First request at t=0.
    Useful for profiling where you want deterministic, isolated requests.
    """
    if n <= 0:
        return []
    gap = 1.0 / arrival_rate
    return [round(i * gap, 3) for i in range(n)]


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_workload(
    total_requests: int,
    long_fraction: float,
    arrival_rate: float,
    seed: int,
    model_path: str,
    endpoint: str,
    arrival_mode: str = "poisson",
) -> dict:
    rng = np.random.default_rng(seed)

    n_long = round(total_requests * long_fraction)
    n_short = total_requests - n_long

    # Assign class labels and shuffle so short/long are intermixed
    labels = ["short"] * n_short + ["long"] * n_long
    indices = rng.permutation(total_requests)
    ordered_labels = [labels[i] for i in indices]

    # Generate arrival times
    if arrival_mode == "uniform":
        offsets = uniform_offsets(total_requests, arrival_rate)
    else:
        offsets = sample_poisson_offsets(total_requests, arrival_rate, rng)

    # Build request list
    requests = []
    for i, label in enumerate(ordered_labels):
        if label == "short":
            topic = SHORT_TOPICS[int(rng.integers(0, len(SHORT_TOPICS)))]
            prompt_text = PROMPT_TEMPLATES["short"].format(topic=topic)
            max_new_tokens = 128
            prompt_type = "chat"
        else:
            topic = REASONING_TOPICS[int(rng.integers(0, len(REASONING_TOPICS)))]
            prompt_text = PROMPT_TEMPLATES["long"].format(topic=topic)
            max_new_tokens = 32768
            prompt_type = "reasoning"

        requests.append({
            "request_id": f"req_{i:04d}",
            "request_order": i,
            "class_label": label,
            "prompt_text": prompt_text,
            "submit_time_offset": offsets[i],
            "max_new_tokens": max_new_tokens,
            "prompt_type": prompt_type,
        })

    # Workload name for identification
    if long_fraction == 0.0:
        workload_name = "uniform_short"
    elif long_fraction == 1.0:
        workload_name = "uniform_long"
    else:
        pct = int(round(long_fraction * 100))
        workload_name = f"bimodal_{pct}pct_long"

    duration = max(r["submit_time_offset"] for r in requests) if requests else 0.0

    return {
        "workload_name": workload_name,
        "model_path": model_path,
        "endpoint": endpoint,
        "seed": seed,

        # Workload parameters (for reproducibility and analysis)
        "params": {
            "total_requests": total_requests,
            "long_fraction": long_fraction,
            "arrival_rate_rps": arrival_rate,
            "n_short": n_short,
            "n_long": n_long,
        },

        # Summary statistics (for quick inspection)
        "summary": {
            "duration_seconds": round(duration, 1),
            "mean_inter_arrival_seconds": round(1.0 / arrival_rate, 3),
        },

        "requests": requests,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_summary(workload: dict) -> None:
    p = workload["params"]
    s = workload["summary"]
    print(f"Workload: {workload['workload_name']}")
    print(f"  Requests:       {p['total_requests']} ({p['n_short']} short + {p['n_long']} long)")
    print(f"  Arrival rate:   {p['arrival_rate_rps']} req/s (mean gap: {s['mean_inter_arrival_seconds']}s)")
    print(f"  Duration:       ~{s['duration_seconds']}s")

    # Show first 10 requests
    print(f"\n  {'#':<5} {'t(s)':<10} {'class':<8} {'max_tokens':<10}")
    print(f"  {'-'*5} {'-'*10} {'-'*8} {'-'*10}")
    for req in workload["requests"][:10]:
        print(
            f"  {req['request_order']:<5} "
            f"{req['submit_time_offset']:<10.3f} "
            f"{req['class_label']:<8} "
            f"{req['max_new_tokens']:<10}"
        )
    if len(workload["requests"]) > 10:
        print(f"  ... ({len(workload['requests']) - 10} more)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a workload JSON for SGLang admission-control experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Light load, 20% reasoning
  python generate_workload.py --arrival-rate 0.5 --long-fraction 0.2

  # Heavy load, 40% reasoning
  python generate_workload.py --arrival-rate 2.0 --long-fraction 0.4

  # Uniform baseline (no reasoning requests)
  python generate_workload.py --arrival-rate 2.0 --long-fraction 0.0

  # Sweep load factor: generate multiple workloads
  for rate in 0.5 1.0 1.5 2.0 2.5 3.0; do
    python generate_workload.py --arrival-rate $rate --long-fraction 0.2 \\
      --output workloads/bimodal_20_rate_${rate}.json
  done
        """,
    )
    parser.add_argument("--total-requests", type=int, default=200,
                        help="Total number of requests (default: 200)")
    parser.add_argument("--long-fraction", type=float, default=0.2,
                        help="Fraction of long/reasoning requests [0.0, 1.0] (default: 0.2)")
    parser.add_argument("--arrival-rate", type=float, required=True,
                        help="Mean arrival rate in requests/second. "
                             "Use load_calculator.py to compute this from profiled values.")
    parser.add_argument("--arrival-mode", choices=["poisson", "uniform"], default="poisson",
                        help="Arrival process: poisson (random, default) or uniform (deterministic equal spacing).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--model-path", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model path (metadata only)")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:30000",
                        help="Server endpoint (metadata only)")
    parser.add_argument("--output", type=str, default="workloads/data/workload.json",
                        help="Output JSON file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workload = generate_workload(
        total_requests=args.total_requests,
        long_fraction=args.long_fraction,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
        model_path=args.model_path,
        endpoint=args.endpoint,
        arrival_mode=args.arrival_mode,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(workload, f, indent=2)

    print_summary(workload)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()