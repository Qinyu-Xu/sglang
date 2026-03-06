"""Run initialization and request logging for experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

CONFIG_FILE = "config.json"
REQUESTS_FILE = "requests.jsonl"
SUMMARY_FILE = "summary.json"

REQUIRED_REQUEST_FIELDS = (
    "request_id",
    "class_label",
    "submit_ts",
    "first_token_ts",
    "finish_ts",
    "status",
    "prompt_len",
    "output_len",
    "error_msg",
)


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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_dir) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)

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
        return {k: request_record.get(k) for k in REQUIRED_REQUEST_FIELDS}
