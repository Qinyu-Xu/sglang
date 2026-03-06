"""Run controlled generated workloads and persist experiment logs."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

try:
    from .logger import ExperimentLogger
    from .summarize import generate_summary_for_run
    from .workload_generator import WorkloadType, generate_workload
except ImportError:  # pragma: no cover - allows direct script execution
    from logger import ExperimentLogger
    from summarize import generate_summary_for_run
    from workload_generator import WorkloadType, generate_workload

REQUEST_PLAN_FILE = "request_plan.jsonl"


def run_controlled_workload(
    workload_type: WorkloadType,
    num_requests: int = 12,
    inter_arrival_sec: float = 0.07,
    seed: int = 0,
    base_dir: str | Path = "results",
) -> Path:
    """Generate a controlled workload, log simulated execution, and summarize."""
    workload_name = f"baseline_{workload_type}_v1"
    planned_requests = generate_workload(
        workload_type=workload_type,
        num_requests=num_requests,
        inter_arrival_sec=inter_arrival_sec,
        seed=seed,
    )

    logger = ExperimentLogger.init_run(base_dir=base_dir)
    logger.write_config(
        {
            "workload_name": workload_name,
            "workload_type": workload_type,
            "num_requests": num_requests,
            "inter_arrival_sec": inter_arrival_sec,
            "seed": seed,
            "runner": "controlled_workload_runner",
        }
    )
    _write_request_plan(logger.run_dir / REQUEST_PLAN_FILE, planned_requests)

    rng = random.Random(seed + 1)
    run_start_ts = time.time()
    for idx, planned in enumerate(planned_requests):
        submit_ts = run_start_ts + planned.submit_time_offset
        ttft_delta = _sample_ttft_seconds(planned.class_label, rng)
        decode_delta = _sample_decode_seconds(
            planned.class_label, planned.max_new_tokens, rng
        )
        first_token_ts = submit_ts + ttft_delta
        finish_ts = first_token_ts + decode_delta

        logger.log_request(
            {
                "request_id": planned.request_id,
                "request_order": idx,
                "workload_name": workload_name,
                "class_label": planned.class_label,
                "submit_ts": submit_ts,
                "first_token_ts": first_token_ts,
                "finish_ts": finish_ts,
                "status": "success",
                "prompt_len": _estimate_prompt_len(planned.prompt_text),
                "output_len": planned.max_new_tokens,
                "error_msg": "",
                "target_output_len": planned.max_new_tokens,
                "max_new_tokens_per_request": planned.max_new_tokens,
                "prompt_type": (
                    "reasoning"
                    if planned.class_label == "long_reasoning_req"
                    else "chat"
                ),
            }
        )

    generate_summary_for_run(logger.run_dir)
    return logger.run_dir


def _write_request_plan(path: Path, planned_requests: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for req in planned_requests:
            f.write(json.dumps(req.to_dict(), ensure_ascii=True) + "\n")


def _sample_ttft_seconds(class_label: str, rng: random.Random) -> float:
    if class_label == "short_req":
        return rng.uniform(0.03, 0.12)
    if class_label == "long_reasoning_req":
        return rng.uniform(0.18, 0.45)
    raise ValueError(f"Unsupported class_label: {class_label}")


def _sample_decode_seconds(
    class_label: str, max_new_tokens: int, rng: random.Random
) -> float:
    if class_label == "short_req":
        token_rate = rng.uniform(180.0, 260.0)
    elif class_label == "long_reasoning_req":
        token_rate = rng.uniform(80.0, 140.0)
    else:
        raise ValueError(f"Unsupported class_label: {class_label}")
    return max_new_tokens / token_rate


def _estimate_prompt_len(prompt_text: str) -> int:
    # Cheap deterministic approximation good enough for baseline experiments.
    return max(1, len(prompt_text) // 4)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a controlled workload experiment.")
    parser.add_argument(
        "--workload-type",
        choices=("short_only", "long_only", "mixed"),
        default="mixed",
        help="Workload composition to generate.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=12,
        help="Number of requests in the generated workload.",
    )
    parser.add_argument(
        "--inter-arrival-sec",
        type=float,
        default=0.07,
        help="Fixed submit-time spacing between consecutive requests.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--base-dir",
        default="results",
        help="Directory where run_YYYYMMDD_HHMMSS folders are created.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    run_dir = run_controlled_workload(
        workload_type=args.workload_type,
        num_requests=args.num_requests,
        inter_arrival_sec=args.inter_arrival_sec,
        seed=args.seed,
        base_dir=args.base_dir,
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
