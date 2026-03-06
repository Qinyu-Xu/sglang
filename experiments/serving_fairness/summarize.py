"""Compute and write summary metrics from requests.jsonl."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .logger import REQUESTS_FILE, SUMMARY_FILE
except ImportError:  # pragma: no cover - allows direct script execution
    from logger import REQUESTS_FILE, SUMMARY_FILE


def read_requests_jsonl(requests_path: str | Path) -> list[dict[str, Any]]:
    """Read newline-delimited request records from a jsonl file."""
    path = Path(requests_path)
    records: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Requests file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Expected object on line {line_number} in {path}, got {type(loaded)!r}"
                )
            records.append(loaded)
    return records


def compute_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute overall and per-class summary metrics."""
    items = list(records)
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for record in items:
        class_label = str(record.get("class_label", "unknown"))
        by_class[class_label].append(record)

    summary: dict[str, Any] = _compute_group_metrics(items)
    summary["by_class_label"] = {
        class_label: _compute_group_metrics(group_items)
        for class_label, group_items in sorted(by_class.items(), key=lambda x: x[0])
    }
    return summary


def write_summary_json(summary: Mapping[str, Any], summary_path: str | Path) -> None:
    """Write summary dictionary to summary.json."""
    path = Path(summary_path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def generate_summary_for_run(run_dir: str | Path) -> dict[str, Any]:
    """Read requests.jsonl in run_dir, compute metrics, and write summary.json."""
    run_path = Path(run_dir)
    requests_path = run_path / REQUESTS_FILE
    summary_path = run_path / SUMMARY_FILE

    records = read_requests_jsonl(requests_path)
    summary = compute_summary(records)
    write_summary_json(summary, summary_path)
    return summary


def _compute_group_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(records)
    total_count = len(items)
    success_count = sum(1 for r in items if _normalized_status(r) == "success")
    timeout_count = sum(1 for r in items if _normalized_status(r) == "timeout")
    error_only_count = sum(1 for r in items if _normalized_status(r) == "error")
    error_count = error_only_count + timeout_count
    ttft_values = _compute_ttft_values(items)
    completion_values = _compute_completion_values(items)

    return {
        "total_request_count": total_count,
        "success_count": success_count,
        "error_count": error_count,
        "timeout_count": timeout_count,
        "ttft_seconds": _percentile_block(ttft_values),
        "completion_latency_seconds": _percentile_block(completion_values),
    }


def _compute_ttft_values(records: Iterable[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        submit_ts = _as_float(record.get("submit_ts"))
        first_token_ts = _as_float(record.get("first_token_ts"))
        if submit_ts is None or first_token_ts is None:
            continue
        if first_token_ts < submit_ts:
            continue
        values.append(first_token_ts - submit_ts)
    return values


def _compute_completion_values(records: Iterable[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        submit_ts = _as_float(record.get("submit_ts"))
        finish_ts = _as_float(record.get("finish_ts"))
        if submit_ts is None or finish_ts is None:
            continue
        if finish_ts < submit_ts:
            continue
        values.append(finish_ts - submit_ts)
    return values


def _percentile_block(values: list[float]) -> dict[str, float | int | None]:
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "p50": _percentile(sorted_values, 50),
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
    }


def _percentile(sorted_values: list[float], percentile: int) -> float | None:
    """Nearest-rank percentile on a pre-sorted list."""
    if not sorted_values:
        return None
    rank = math.ceil((percentile / 100.0) * len(sorted_values))
    index = min(max(rank - 1, 0), len(sorted_values) - 1)
    return sorted_values[index]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_status(record: Mapping[str, Any]) -> str:
    return str(record.get("status", "")).lower()


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate summary.json from requests.jsonl.")
    parser.add_argument("run_dir", help="Path to a run directory containing requests.jsonl.")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    generate_summary_for_run(args.run_dir)


if __name__ == "__main__":
    main()
