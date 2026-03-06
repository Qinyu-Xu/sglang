"""Controlled workload generation for baseline serving experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

WorkloadType = Literal["short_only", "long_only", "mixed"]


@dataclass(frozen=True)
class WorkloadRequest:
    """A planned request for a controlled experiment workload."""

    request_id: str
    class_label: str
    prompt_text: str
    submit_time_offset: float
    max_new_tokens: int

    def to_dict(self) -> dict[str, str | float | int]:
        return {
            "request_id": self.request_id,
            "class_label": self.class_label,
            "prompt_text": self.prompt_text,
            "submit_time_offset": self.submit_time_offset,
            "max_new_tokens": self.max_new_tokens,
        }


def generate_workload(
    workload_type: WorkloadType,
    num_requests: int = 12,
    inter_arrival_sec: float = 0.07,
    seed: int = 0,
) -> list[WorkloadRequest]:
    """Generate a controlled workload with short, long, or mixed demand."""
    if num_requests <= 0:
        raise ValueError("num_requests must be > 0")
    if inter_arrival_sec < 0:
        raise ValueError("inter_arrival_sec must be >= 0")

    rng = random.Random(seed)
    class_sequence = _build_class_sequence(workload_type, num_requests)

    requests: list[WorkloadRequest] = []
    for idx, class_label in enumerate(class_sequence):
        submit_time_offset = idx * inter_arrival_sec
        max_new_tokens = _sample_max_new_tokens(class_label, rng)
        requests.append(
            WorkloadRequest(
                request_id=f"req_{idx:04d}",
                class_label=class_label,
                prompt_text=_build_prompt_text(class_label, idx),
                submit_time_offset=submit_time_offset,
                max_new_tokens=max_new_tokens,
            )
        )
    return requests


def _build_class_sequence(workload_type: WorkloadType, num_requests: int) -> list[str]:
    if workload_type == "short_only":
        return ["short_req"] * num_requests
    if workload_type == "long_only":
        return ["long_reasoning_req"] * num_requests
    if workload_type == "mixed":
        # Deterministic 1:1 alternation keeps baseline comparisons simple.
        return [
            "short_req" if idx % 2 == 0 else "long_reasoning_req"
            for idx in range(num_requests)
        ]
    raise ValueError(f"Unsupported workload_type: {workload_type}")


def _sample_max_new_tokens(class_label: str, rng: random.Random) -> int:
    if class_label == "short_req":
        return rng.randint(24, 96)
    if class_label == "long_reasoning_req":
        return rng.randint(256, 512)
    raise ValueError(f"Unsupported class_label: {class_label}")


def _build_prompt_text(class_label: str, request_index: int) -> str:
    if class_label == "short_req":
        return (
            f"[short:{request_index}] Give a concise 1-2 sentence response "
            "to: What is the key idea of batching in LLM serving?"
        )
    if class_label == "long_reasoning_req":
        return (
            f"[long_reasoning:{request_index}] Solve step-by-step: compare FIFO and "
            "shortest-job-first scheduling for mixed-length LLM requests."
        )
    raise ValueError(f"Unsupported class_label: {class_label}")
