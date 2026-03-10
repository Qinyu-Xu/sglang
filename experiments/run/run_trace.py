"""
Replay a workload spec against an OpenAI-compatible endpoint and log request traces.

The client submits each request at its prescribed submit_time_offset and
does NOT enforce any concurrency cap. The number of concurrent in-flight
requests is an emergent property of arrival rate × service time — which is
exactly what we want to measure.

Usage:
    python replay_workload.py --workload-spec data/frac20_rate2p0.json
    python replay_workload.py --workload-spec data/frac20_rate2p0.json --timeout 600
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from logger import DEFAULT_RESULTS_DIR, ExperimentLogger

try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)
except ImportError:

    def _progress(iterable, total=None, desc=None, **kwargs):
        if desc:
            print(f"{desc} ...")
        return iterable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRequest:
    """One request entry from a workload specification."""

    request_id: str
    request_order: int
    workload_name: str
    class_label: str
    prompt_text: str
    submit_time_offset: float
    max_new_tokens: int
    prompt_type: str


# ---------------------------------------------------------------------------
# Core replay logic
# ---------------------------------------------------------------------------


def replay_trace(
    workload_spec_path: str | Path,
    endpoint: str = "http://127.0.0.1:30000",
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    timeout: int = 600,
    base_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """
    Replay workload requests against endpoint.

    There is NO client-side concurrency cap. Every request is dispatched
    at its submit_time_offset by its own thread. The number of in-flight
    requests is determined entirely by the arrival rate and server-side
    service times — which is what we want to observe.
    """
    workload_name, requests = _read_workload_spec(workload_spec_path)

    logger = ExperimentLogger.init_run(base_dir=base_dir)
    logger.write_config(
        {
            "workload_name": workload_name,
            "workload_spec_path": str(workload_spec_path),
            "endpoint": endpoint,
            "model_path": model_path,
            "timeout_seconds": timeout,
            "total_requests": len(requests),
        }
    )

    requests_sorted = sorted(
        requests, key=lambda r: (r.submit_time_offset, r.request_order)
    )

    total = len(requests_sorted)
    result_q: queue.Queue[dict[str, Any]] = queue.Queue()
    inflight = threading.Semaphore(0)  # tracks completions

    def _worker(req: TraceRequest) -> None:
        """Wait until submit time, send request, enqueue result."""
        _sleep_until_offset(run_start_ts, req.submit_time_offset)
        record = _send_one_request(req, endpoint, model_path, timeout)
        result_q.put(record)

    # Launch one thread per request — they are I/O-bound so this is fine
    run_start_ts = time.time()
    threads: list[threading.Thread] = []
    for req in requests_sorted:
        t = threading.Thread(target=_worker, args=(req,), daemon=True)
        t.start()
        threads.append(t)

    # Collect results as they complete
    complete_bar = _progress(
        range(total),
        total=total,
        desc=f"[{workload_name}] completing",
        unit="req",
    )
    for _ in complete_bar:
        record = result_q.get()
        logger.log_request(record)

    # Wait for all threads to finish (they should be done since we got all results)
    for t in threads:
        t.join(timeout=5)

    run_duration = time.time() - run_start_ts
    print(f"\n[{workload_name}] Finished in {run_duration:.1f}s")
    return logger.run_dir


# ---------------------------------------------------------------------------
# Workload spec reader
# ---------------------------------------------------------------------------


def _read_workload_spec(path: str | Path) -> tuple[str, list[TraceRequest]]:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    if not isinstance(spec, dict):
        raise ValueError("workload spec must be a JSON object")
    workload_name = str(spec.get("workload_name", "unnamed_workload"))
    raw_requests = spec.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("workload spec must include a non-empty 'requests' list")

    requests: list[TraceRequest] = []
    for idx, raw in enumerate(raw_requests):
        if not isinstance(raw, dict):
            raise ValueError(f"Request #{idx} must be a JSON object")

        request_order = _as_int(raw.get("request_order", idx), name="request_order")
        request_id = str(raw.get("request_id", f"req_{idx:06d}"))
        class_label = str(raw.get("class_label", "unknown"))
        prompt_text = str(raw.get("prompt_text", ""))
        submit_time_offset = _as_float(
            raw.get("submit_time_offset"), name="submit_time_offset"
        )
        max_new_tokens = _as_int(
            raw.get("max_new_tokens", 64), name="max_new_tokens"
        )
        prompt_type = str(
            raw.get("prompt_type", "reasoning" if "long" in class_label else "chat")
        )

        requests.append(
            TraceRequest(
                request_id=request_id,
                request_order=request_order,
                workload_name=workload_name,
                class_label=class_label,
                prompt_text=prompt_text,
                submit_time_offset=submit_time_offset,
                max_new_tokens=max_new_tokens,
                prompt_type=prompt_type,
            )
        )
    return workload_name, requests


# ---------------------------------------------------------------------------
# HTTP request sender
# ---------------------------------------------------------------------------


def _send_one_request(
    request: TraceRequest,
    endpoint: str,
    model_path: str,
    timeout: int,
) -> dict[str, Any]:
    submit_ts = time.time()
    first_token_ts: float | None = None
    finish_ts: float | None = None
    output_token_count = 0
    output_text_chunks: list[str] = []
    status = "success"
    error_msg = ""

    payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": request.prompt_text}],
        "max_tokens": request.max_new_tokens,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    request_url = endpoint.rstrip("/") + "/v1/chat/completions"
    http_request = urllib.request.Request(
        request_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_line = line[6:]
                if data_line == "[DONE]":
                    break
                chunk = json.loads(data_line)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token_piece = delta.get("content")
                if token_piece:
                    output_text_chunks.append(str(token_piece))
                    output_token_count += 1
                    if first_token_ts is None:
                        first_token_ts = time.time()
            finish_ts = time.time()
    except urllib.error.HTTPError as exc:
        status = "error"
        error_msg = f"HTTPError {exc.code}: {exc.reason}"
        finish_ts = time.time()
    except (TimeoutError, OSError) as exc:
        status = "timeout"
        error_msg = f"timeout after {timeout}s"
        finish_ts = time.time()
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        status = "error"
        error_msg = str(exc)
        finish_ts = time.time()

    # Compute derived metrics
    ttft = (first_token_ts - submit_ts) if first_token_ts else None
    completion_latency = (finish_ts - submit_ts) if finish_ts else None
    generation_time = (
        (finish_ts - first_token_ts) if (first_token_ts and finish_ts) else None
    )

    return {
        # Identity
        "request_id": request.request_id,
        "request_order": request.request_order,
        "workload_name": request.workload_name,
        "class_label": request.class_label,
        "prompt_type": request.prompt_type,
        # Timestamps (absolute, for timeline analysis)
        "submit_ts": submit_ts,
        "first_token_ts": first_token_ts,
        "finish_ts": finish_ts,
        # Latency metrics (seconds)
        "ttft": ttft,
        "completion_latency": completion_latency,
        "generation_time": generation_time,
        # Token counts
        "prompt_len": len(request.prompt_text.split()),
        "output_len": output_token_count,
        "max_new_tokens": request.max_new_tokens,
        # Status
        "status": status,
        "error_msg": error_msg,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sleep_until_offset(run_start_ts: float, submit_time_offset: float) -> None:
    """Sleep until the target wall-clock time."""
    target_ts = run_start_ts + submit_time_offset
    remaining = target_ts - time.time()
    if remaining > 0:
        time.sleep(remaining)


def _as_int(value: Any, name: str) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for '{name}': {value!r}") from exc
    if ivalue < 0:
        raise ValueError(f"'{name}' must be >= 0")
    return ivalue


def _as_float(value: Any, name: str) -> float:
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for '{name}': {value!r}") from exc
    if fvalue < 0:
        raise ValueError(f"'{name}' must be >= 0")
    return fvalue


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay workload trace against SGLang endpoint."
    )
    parser.add_argument(
        "--workload-spec", required=True, help="Path to workload JSON file."
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:30000",
        help="OpenAI-compatible endpoint base URL (default: http://127.0.0.1:30000).",
    )
    parser.add_argument(
        "--model-path",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model id passed in OpenAI requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-request timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Base directory for run output folders.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    run_dir = replay_trace(
        workload_spec_path=args.workload_spec,
        endpoint=args.endpoint,
        model_path=args.model_path,
        timeout=args.timeout,
        base_dir=args.base_dir,
    )
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()