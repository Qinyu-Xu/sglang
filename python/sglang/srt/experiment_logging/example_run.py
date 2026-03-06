"""Example script that runs a controlled mixed workload experiment."""

from __future__ import annotations

try:
    from .workload_runner import run_controlled_workload
except ImportError:  # pragma: no cover - allows direct script execution
    from workload_runner import run_controlled_workload


def main() -> None:
    run_dir = run_controlled_workload(
        workload_type="mixed",
        num_requests=10,
        inter_arrival_sec=0.07,
        seed=7,
        base_dir="results",
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
