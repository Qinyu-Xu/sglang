"""Run initialization and request logging for experiments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

CONFIG_FILE = "config.json"
REQUESTS_FILE = "requests.jsonl"
SUMMARY_FILE = "summary.json"

REQUIRED_REQUEST_FIELDS = (
    "request_id",
    "request_order",
    "workload_name",
    "class_label",
    "submit_ts",
    "first_token_ts",
    "finish_ts",
    "status",
    "prompt_len",
    "output_len",
    "error_msg",
)

OPTIONAL_REQUEST_FIELDS = (
    "target_output_len",
    "max_new_tokens_per_request",
    "prompt_type",
)

VALID_STATUSES = {"success", "error", "timeout"}


@dataclass(frozen=True)
class ExperimentLogger:
    """Logger that writes experiment artifacts to a single run directory."""

    run_dir: Path
    config_path: Path
    requests_path: Path
    summary_path: Path

    @classmethod
    def init_run(cls, base_dir: str | Path = "results") -> "ExperimentLogger":
        """Create a timestamped run directory under the base results directory."""
        run_dir: Path | None = None
        for _ in range(3):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate = Path(base_dir) / f"run_{timestamp}"
            try:
                candidate.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                # Retry with a fresh timestamp if two runs start in the same second.
                time.sleep(1.05)
                continue
            run_dir = candidate
            break
        if run_dir is None:
            raise RuntimeError("Failed to create a unique run directory after retries")

        logger = cls(
            run_dir=run_dir,
            config_path=run_dir / CONFIG_FILE,
            requests_path=run_dir / REQUESTS_FILE,
            summary_path=run_dir / SUMMARY_FILE,
        )

        # Create an empty JSONL file up front for easier inspection.
        logger.requests_path.touch(exist_ok=False)
        return logger

    def write_config(self, config: Mapping[str, Any]) -> None:
        """Write run configuration to config.json."""
        self._write_json_file(self.config_path, dict(config))

    def log_request(self, request_record: Mapping[str, Any]) -> None:
        """Append one request record as a JSON line."""
        normalized = self._normalize_request_record(request_record)
        with self.requests_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(normalized, ensure_ascii=True) + "\n")

    @staticmethod
    def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")

    @staticmethod
    def _normalize_request_record(request_record: Mapping[str, Any]) -> dict[str, Any]:
        missing_fields = [k for k in REQUIRED_REQUEST_FIELDS if k not in request_record]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Request record is missing required fields: {missing}")

        normalized: dict[str, Any] = {
            k: request_record.get(k) for k in REQUIRED_REQUEST_FIELDS
        }
        normalized["status"] = str(normalized["status"]).lower()
        if normalized["status"] not in VALID_STATUSES:
            valid = ", ".join(sorted(VALID_STATUSES))
            raise ValueError(
                f"Invalid request status '{normalized['status']}'. Expected one of: {valid}"
            )

        for optional_key in OPTIONAL_REQUEST_FIELDS:
            normalized[optional_key] = request_record.get(optional_key)

        _validate_timestamp_order(normalized)
        return normalized


def _validate_timestamp_order(request_record: Mapping[str, Any]) -> None:
    """Validate timestamp monotonicity when values are present."""
    submit_ts = _as_float(request_record.get("submit_ts"))
    first_token_ts = _as_float(request_record.get("first_token_ts"))
    finish_ts = _as_float(request_record.get("finish_ts"))

    if submit_ts is not None and first_token_ts is not None and first_token_ts < submit_ts:
        raise ValueError("Invalid timestamps: first_token_ts is earlier than submit_ts")
    if submit_ts is not None and finish_ts is not None and finish_ts < submit_ts:
        raise ValueError("Invalid timestamps: finish_ts is earlier than submit_ts")
    if (
        first_token_ts is not None
        and finish_ts is not None
        and finish_ts < first_token_ts
    ):
        raise ValueError("Invalid timestamps: finish_ts is earlier than first_token_ts")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
