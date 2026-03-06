"""Small example that logs fake requests and generates summary.json."""

from __future__ import annotations

import random
import time
from pathlib import Path

try:
    from .logger import ExperimentLogger
    from .summarize import generate_summary_for_run
except ImportError:  # pragma: no cover - allows direct script execution
    from logger import ExperimentLogger
    from summarize import generate_summary_for_run


def main() -> None:
    random.seed(7)
    logger = ExperimentLogger.init_run(base_dir=Path("results"))

    config = {
        "model": "fake-llm-8b",
        "dataset": "synthetic_prompts_v1",
        "max_new_tokens": 128,
        "concurrency": 4,
    }
    logger.write_config(config)

    class_labels = ["short_prompt", "long_prompt", "tool_call"]

    now = time.time()
    for idx in range(8):
        submit_ts = now + idx * 0.07
        is_error = idx in (3, 6)
        first_token_delta = random.uniform(0.03, 0.35) if not is_error else None
        completion_delta = random.uniform(0.25, 1.25) if not is_error else None

        logger.log_request(
            {
                "request_id": f"req_{idx:03d}",
                "class_label": class_labels[idx % len(class_labels)],
                "submit_ts": submit_ts,
                "first_token_ts": (
                    submit_ts + first_token_delta
                    if first_token_delta is not None
                    else None
                ),
                "finish_ts": (
                    submit_ts + completion_delta if completion_delta is not None else None
                ),
                "status": "error" if is_error else "success",
                "prompt_len": random.randint(16, 1024),
                "output_len": 0 if is_error else random.randint(24, 256),
                "error_msg": "mock timeout" if is_error else "",
            }
        )

    summary = generate_summary_for_run(logger.run_dir)
    print(f"Run directory: {logger.run_dir}")
    print(
        "Summary counts:",
        summary["total_request_count"],
        summary["success_count"],
        summary["error_count"],
    )


if __name__ == "__main__":
    main()
