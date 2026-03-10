#!/usr/bin/env python3
"""Step 3 — Compute calibrated arrival rates for each workload.

Given the median service times measured in step 1 (or from the sweep in
step 2), computes the arrival rate that targets a given utilization for
each workload mix.

Formula
-------
Effective load on the server for a mixed workload:

    rho = arrival_rate * (short_frac * svc_short + long_frac * svc_long)

Solving for arrival_rate at target utilization U:

    arrival_rate = U / (short_frac * svc_short + long_frac * svc_long)

Usage (from service_times.json produced by step 1):
    python 03_compute_target_rates.py --from-file service_times.json

Usage (manual):
    python 03_compute_target_rates.py \
        --short-svc-time 1.2 \
        --long-svc-time  18.5 \
        --target-util    0.7
"""

from __future__ import annotations

import argparse
import json
import sys


WORKLOAD_MIXES = {
    "short_only": {"short_frac": 1.0, "long_frac": 0.0},
    "mixed_20":   {"short_frac": 0.8, "long_frac": 0.2},
    "mixed_40":   {"short_frac": 0.6, "long_frac": 0.4},
    "mixed_60":   {"short_frac": 0.4, "long_frac": 0.6},
    "long_only":  {"short_frac": 0.0, "long_frac": 1.0},
}


def compute_rates(short_svc: float, long_svc: float,
                  target_util: float) -> dict[str, float]:
    rates: dict[str, float] = {}
    for name, mix in WORKLOAD_MIXES.items():
        mean_svc = mix["short_frac"] * short_svc + mix["long_frac"] * long_svc
        rate = target_util / mean_svc
        rates[name] = round(rate, 4)
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute calibrated arrival rates for each workload mix."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-file", metavar="service_times.json",
                     help="JSON output from 01_measure_service_time.py")
    src.add_argument("--short-svc-time", type=float,
                     help="Median completion latency for a short_req (seconds).")

    parser.add_argument("--long-svc-time", type=float,
                        help="Median completion latency for a long_reasoning_req (seconds)."
                             " Required when using --short-svc-time.")
    parser.add_argument("--target-util", type=float, default=0.7,
                        help="Target server utilization [0.0, 1.0]. Default: 0.7")
    parser.add_argument("--output", default=None,
                        help="Optional: write rates JSON to this file.")
    args = parser.parse_args()

    # -- Load service times -----------------------------------------------
    if args.from_file:
        with open(args.from_file) as f:
            svc = json.load(f)
        short_svc = svc.get("short_req", {}).get("completion_median")
        long_svc  = svc.get("long_reasoning_req", {}).get("completion_median")
        if short_svc is None or long_svc is None:
            print("ERROR: service_times.json missing completion_median for one or both classes.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Loaded from {args.from_file}:")
    else:
        if args.long_svc_time is None:
            print("ERROR: --long-svc-time required when using --short-svc-time.",
                  file=sys.stderr)
            sys.exit(1)
        short_svc = args.short_svc_time
        long_svc  = args.long_svc_time

    print(f"  short_req completion_median : {short_svc:.3f}s")
    print(f"  long_req  completion_median : {long_svc:.3f}s")
    print(f"  target utilization          : {args.target_util:.0%}")
    print()

    rates = compute_rates(short_svc, long_svc, args.target_util)

    # -- Print table -------------------------------------------------------
    print(f"{'Workload':<14}  {'short_frac':>10}  {'long_frac':>9}  "
          f"{'mean_svc(s)':>11}  {'arrival_rate':>12}")
    print("-" * 65)
    for name, mix in WORKLOAD_MIXES.items():
        mean_svc = mix["short_frac"] * short_svc + mix["long_frac"] * long_svc
        rate = rates[name]
        print(f"{name:<14}  {mix['short_frac']:>10.1f}  {mix['long_frac']:>9.1f}  "
              f"{mean_svc:>11.3f}  {rate:>12.4f}")

    print()
    print("Shell export for 04_regen_calibrated_workloads.sh:")
    print()
    for name, rate in rates.items():
        env_name = f"RATE_{name.upper()}"
        print(f"  export {env_name}={rate}")

    # -- Optionally write JSON ---------------------------------------------
    if args.output:
        out = {
            "short_svc_time": short_svc,
            "long_svc_time":  long_svc,
            "target_util":    args.target_util,
            "rates":          rates,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nRates saved to: {args.output}")


if __name__ == "__main__":
    main()
