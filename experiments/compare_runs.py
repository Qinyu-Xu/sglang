"""Compare multiple experiment runs and compute per-class fairness slowdowns."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from sglang.experiments.run.summarize import compute_summary, read_requests_jsonl


@dataclass(frozen=True)
class ClassMetrics:
    """Per-class metrics for one run."""

    class_label: str
    total_request_count: int
    success_count: int
    timeout_count: int
    error_count: int
    ttft_p50: float | None
    ttft_p95: float | None
    ttft_p99: float | None
    completion_p50: float | None
    completion_p95: float | None
    completion_p99: float | None


@dataclass(frozen=True)
class RunClassRow:
    """One CSV/Markdown row for run-by-class comparison."""

    run_dir: str
    workload_name: str
    class_label: str
    total_request_count: int
    success_count: int
    timeout_count: int
    error_count: int
    ttft_p50: float | None
    ttft_p95: float | None
    ttft_p99: float | None
    completion_p50: float | None
    completion_p95: float | None
    completion_p99: float | None
    baseline_run_dir: str | None
    baseline_completion_p50: float | None
    slowdown_vs_baseline: float | None


def compare_runs(run_dirs: Iterable[str | Path]) -> list[RunClassRow]:
    """Compute run/class table with slowdown relative to class-only baselines."""
    run_paths = [Path(p) for p in run_dirs]
    run_summaries = [_load_run_summary(path) for path in run_paths]
    baseline_by_class = _pick_class_only_baselines(run_summaries)

    rows: list[RunClassRow] = []
    for run_path, workload_name, class_metrics in run_summaries:
        for metric in class_metrics:
            baseline = baseline_by_class.get(metric.class_label)
            baseline_run_dir = baseline[0] if baseline else None
            baseline_p50 = baseline[1].completion_p50 if baseline else None
            slowdown = _safe_ratio(metric.completion_p50, baseline_p50)
            rows.append(
                RunClassRow(
                    run_dir=str(run_path),
                    workload_name=workload_name,
                    class_label=metric.class_label,
                    total_request_count=metric.total_request_count,
                    success_count=metric.success_count,
                    timeout_count=metric.timeout_count,
                    error_count=metric.error_count,
                    ttft_p50=metric.ttft_p50,
                    ttft_p95=metric.ttft_p95,
                    ttft_p99=metric.ttft_p99,
                    completion_p50=metric.completion_p50,
                    completion_p95=metric.completion_p95,
                    completion_p99=metric.completion_p99,
                    baseline_run_dir=baseline_run_dir,
                    baseline_completion_p50=baseline_p50,
                    slowdown_vs_baseline=slowdown,
                )
            )
    return sorted(rows, key=lambda r: (r.workload_name, r.class_label, r.run_dir))


def _load_run_summary(run_dir: Path) -> tuple[Path, str, list[ClassMetrics]]:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    requests_path = run_dir / "requests.jsonl"

    workload_name = run_dir.name
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if isinstance(config, dict):
            workload_name = str(config.get("workload_name", workload_name))

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    elif requests_path.exists():
        records = read_requests_jsonl(requests_path)
        summary = compute_summary(records)
    else:
        raise FileNotFoundError(
            f"Run {run_dir} has neither summary.json nor requests.jsonl"
        )

    by_class = summary.get("by_class_label", {})
    class_metrics: list[ClassMetrics] = []
    for class_label, payload in sorted(by_class.items(), key=lambda x: str(x[0])):
        if not isinstance(payload, Mapping):
            continue
        ttft = payload.get("ttft_seconds", {})
        comp = payload.get("completion_latency_seconds", {})
        class_metrics.append(
            ClassMetrics(
                class_label=str(class_label),
                total_request_count=int(payload.get("total_request_count", 0)),
                success_count=int(payload.get("success_count", 0)),
                timeout_count=int(payload.get("timeout_count", 0)),
                error_count=int(payload.get("error_count", 0)),
                ttft_p50=_as_optional_float(ttft, "p50"),
                ttft_p95=_as_optional_float(ttft, "p95"),
                ttft_p99=_as_optional_float(ttft, "p99"),
                completion_p50=_as_optional_float(comp, "p50"),
                completion_p95=_as_optional_float(comp, "p95"),
                completion_p99=_as_optional_float(comp, "p99"),
            )
        )
    return run_dir, workload_name, class_metrics


def _pick_class_only_baselines(
    run_summaries: list[tuple[Path, str, list[ClassMetrics]]],
) -> dict[str, tuple[str, ClassMetrics]]:
    baselines: dict[str, tuple[str, ClassMetrics]] = {}
    for run_path, _workload_name, class_metrics in run_summaries:
        if len(class_metrics) != 1:
            continue
        only = class_metrics[0]
        current = baselines.get(only.class_label)
        if current is None:
            baselines[only.class_label] = (str(run_path), only)
            continue

        _, prev = current
        prev_key = (prev.success_count, prev.total_request_count)
        new_key = (only.success_count, only.total_request_count)
        if new_key > prev_key:
            baselines[only.class_label] = (str(run_path), only)
    return baselines


def write_csv(rows: Iterable[RunClassRow], output_path: str | Path) -> None:
    """Write comparison rows to CSV."""
    path = Path(output_path)
    fieldnames = [
        "run_dir",
        "workload_name",
        "class_label",
        "total_request_count",
        "success_count",
        "timeout_count",
        "error_count",
        "ttft_p50",
        "ttft_p95",
        "ttft_p99",
        "completion_p50",
        "completion_p95",
        "completion_p99",
        "baseline_run_dir",
        "baseline_completion_p50",
        "slowdown_vs_baseline",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def render_markdown(rows: Iterable[RunClassRow]) -> str:
    """Render comparison rows to a markdown table."""
    headers = [
        "run",
        "workload",
        "class",
        "count",
        "success",
        "timeout",
        "error",
        "ttft_p50",
        "ttft_p95",
        "ttft_p99",
        "comp_p50",
        "comp_p95",
        "comp_p99",
        "slowdown_vs_baseline",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(row.run_dir).name,
                    row.workload_name,
                    row.class_label,
                    str(row.total_request_count),
                    str(row.success_count),
                    str(row.timeout_count),
                    str(row.error_count),
                    _fmt_float(row.ttft_p50),
                    _fmt_float(row.ttft_p95),
                    _fmt_float(row.ttft_p99),
                    _fmt_float(row.completion_p50),
                    _fmt_float(row.completion_p95),
                    _fmt_float(row.completion_p99),
                    _fmt_float(row.slowdown_vs_baseline),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _as_optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare per-class metrics across run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Run directories to compare (e.g. results/run_...).",
    )
    parser.add_argument(
        "--output-csv",
        default="comparison.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional output markdown table path.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    rows = compare_runs(args.run_dirs)
    write_csv(rows, args.output_csv)
    print(f"Wrote CSV: {args.output_csv}")
    markdown = render_markdown(rows)
    if args.output_md:
        Path(args.output_md).write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote Markdown: {args.output_md}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
