#!/usr/bin/env python3
"""Step 1 — Measure isolated service time per request class.

Sends N requests per class one at a time (no concurrency) and records
TTFT + completion latency. Results are printed and saved to service_times.json
for use by 03_compute_target_rates.py.

Usage (server must be running):
    python 01_measure_service_time.py \
        --endpoint http://127.0.0.1:30000 \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --n-per-class 3 \
        --output service_times.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

# One representative prompt per class — short and medium-length
SHORT_PROMPT  = "Give a concise answer in 1-2 sentences: What is a database index?"
LONG_PROMPT   = ("Reason step by step and provide a detailed analysis: "
                 "Explain the time complexity of merge sort and derive it "
                 "from the recurrence relation.")

CLASSES = {
    "short_req": {
        "prompt":         SHORT_PROMPT,
        "max_new_tokens": 128,
        "prompt_type":    "chat",
    },
    "long_reasoning_req": {
        "prompt":         LONG_PROMPT,
        "max_new_tokens": 8192,
        "prompt_type":    "reasoning",
    },
}


def send_request(endpoint: str, model_path: str, prompt: str,
                 max_new_tokens: int, timeout: int = 300) -> dict:
    """Send one streaming chat request. Returns timing dict."""
    payload = {
        "model":      model_path,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "stream":     True,
    }
    data = json.dumps(payload).encode()
    url  = endpoint.rstrip("/") + "/v1/chat/completions"
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    submit_ts       = time.time()
    first_token_ts  = None
    token_count     = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            piece = (chunk.get("choices", [{}])[0]
                         .get("delta", {})
                         .get("content"))
            if piece:
                token_count += 1
                if first_token_ts is None:
                    first_token_ts = time.time()

    finish_ts = time.time()
    return {
        "submit_ts":      submit_ts,
        "first_token_ts": first_token_ts,
        "finish_ts":      finish_ts,
        "ttft":           (first_token_ts - submit_ts) if first_token_ts else None,
        "completion":     finish_ts - submit_ts,
        "token_count":    token_count,
    }


def measure_class(label: str, cfg: dict, endpoint: str, model_path: str,
                  n: int) -> dict:
    print(f"\n  [{label}]  max_new_tokens={cfg['max_new_tokens']}")
    ttfts, completions = [], []

    for i in range(n):
        print(f"    request {i+1}/{n} ... ", end="", flush=True)
        try:
            r = send_request(endpoint, model_path, cfg["prompt"],
                             cfg["max_new_tokens"])
            ttfts.append(r["ttft"])
            completions.append(r["completion"])
            print(f"ttft={r['ttft']:.3f}s  completion={r['completion']:.3f}s"
                  f"  tokens≈{r['token_count']}")
        except Exception as exc:
            print(f"ERROR: {exc}")

    if not completions:
        return {"error": "all requests failed"}

    result = {
        "n":                 n,
        "ttft_median":       statistics.median(ttfts) if ttfts else None,
        "ttft_values":       ttfts,
        "completion_median": statistics.median(completions),
        "completion_values": completions,
        "completion_min":    min(completions),
        "completion_max":    max(completions),
    }

    print(f"    → ttft   median={result['ttft_median']:.3f}s")
    print(f"    → compl  median={result['completion_median']:.3f}s  "
          f"min={result['completion_min']:.3f}s  max={result['completion_max']:.3f}s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure isolated service time per class."
    )
    parser.add_argument("--endpoint",   default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--n-per-class", type=int, default=3,
                        help="Number of isolated requests to send per class.")
    parser.add_argument("--output", default="service_times.json",
                        help="Where to save results (for 03_compute_target_rates.py).")
    args = parser.parse_args()

    print(f"Endpoint:  {args.endpoint}")
    print(f"Model:     {args.model_path}")
    print(f"N/class:   {args.n_per_class}")
    print("\nSending isolated requests (concurrency=1, no queueing)...")

    results: dict[str, dict] = {}
    for label, cfg in CLASSES.items():
        results[label] = measure_class(label, cfg, args.endpoint,
                                       args.model_path, args.n_per_class)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print("\nSummary (use these for 03_compute_target_rates.py):")
    for label, r in results.items():
        if "error" in r:
            print(f"  {label}: ERROR")
        else:
            print(f"  {label}: completion_median={r['completion_median']:.3f}s")


if __name__ == "__main__":
    main()
