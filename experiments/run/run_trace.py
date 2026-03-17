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
from summarize import generate_summary_for_run

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
    model_path: str = "Qwen/Qwen3-8B",
    timeout: int = 600,
    base_dir: str | Path = DEFAULT_RESULTS_DIR,
    record_output: bool = False,
    server_log: str | Path | None = None,
    instant_accept_chat: bool = False,
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
            "instant_accept_chat": instant_accept_chat,
        }
    )

    requests_sorted = sorted(
        requests, key=lambda r: (r.submit_time_offset, r.request_order)
    )

    total = len(requests_sorted)
    result_q: queue.Queue[dict[str, Any]] = queue.Queue()

    def _worker(req: TraceRequest) -> None:
        """Wait until submit time, send request, enqueue result."""
        _sleep_until_offset(run_start_ts, req.submit_time_offset)
        record = _send_one_request(req, endpoint, model_path, timeout, record_output)
        result_q.put(record)

    # Snapshot server log offset before dispatch
    server_log_start = _get_file_size(server_log)

    # Snapshot preemption counter before dispatch
    preemptions_before = _read_preemption_counter(endpoint)

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

    # Snapshot preemption counter after all requests complete
    preemptions_after = _read_preemption_counter(endpoint)
    preemptions = None
    if preemptions_before is not None and preemptions_after is not None:
        preemptions = int(preemptions_after - preemptions_before)
        print(f"\n[{workload_name}] Finished in {run_duration:.1f}s  |  preemptions: {preemptions}")
    else:
        print(f"\n[{workload_name}] Finished in {run_duration:.1f}s  |  preemptions: unavailable")

    # Copy server log slice for this run
    _copy_log_slice(server_log, server_log_start, logger.run_dir / "server_log.txt")

    # Auto-summarize
    summary = generate_summary_for_run(logger.run_dir, preemptions=preemptions)
    print(f"[{workload_name}] Summary written to {logger.run_dir / 'summary.json'}")

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
    record_output: bool = False,
) -> dict[str, Any]:
    submit_ts = time.time()
    first_token_ts: float | None = None
    finish_ts: float | None = None
    output_token_count = 0
    output_text_chunks: list[str] = []
    think_open = False
    status = "success"
    error_msg = ""

    payload: dict[str, Any] = {
        "model": model_path,
        "messages": [{"role": "user", "content": request.prompt_text}],
        "max_tokens": request.max_new_tokens,
        "stream": True,
    }
    # Disable chain-of-thought for short/chat requests to avoid wasting KV budget
    if request.prompt_type == "chat":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
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
                think_piece = delta.get("reasoning_content")
                token_piece = delta.get("content")
                if think_piece:
                    if not think_open:
                        output_text_chunks.append("<think>")
                        think_open = True
                    output_text_chunks.append(str(think_piece))
                    output_token_count += 1
                    if first_token_ts is None:
                        first_token_ts = time.time()
                if token_piece:
                    if think_open:
                        output_text_chunks.append("</think>")
                        think_open = False
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

    # Compute derived metrics — all timing captured above, join is post-latency
    ttft = (first_token_ts - submit_ts) if first_token_ts else None
    completion_latency = (finish_ts - submit_ts) if finish_ts else None
    generation_time = (
        (finish_ts - first_token_ts) if (first_token_ts and finish_ts) else None
    )
    output_text = "".join(output_text_chunks) if record_output else None

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
        # Optional full output (only present when --record-output is set)
        "output_text": output_text,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_file_size(path: str | Path | None) -> int | None:
    """Return current byte size of a file, or None if unavailable."""
    if path is None:
        return None
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def _copy_log_slice(
    log_path: str | Path | None, start_offset: int | None, dest: Path
) -> None:
    """Copy bytes from start_offset to EOF into dest."""
    if log_path is None or start_offset is None:
        return
    try:
        with open(log_path, "rb") as src, open(dest, "wb") as dst:
            src.seek(start_offset)
            while chunk := src.read(65536):
                dst.write(chunk)
    except OSError:
        pass


def _read_preemption_counter(endpoint: str) -> float | None:
    """Read sglang:num_preemption_total from the /metrics Prometheus endpoint."""
    url = endpoint.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace")
                if line.startswith("sglang:num_preemption_total"):
                    return float(line.split()[-1])
    except Exception:
        pass
    return None


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
        default="Qwen/Qwen3-8B",
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
    parser.add_argument(
        "--record-output",
        action="store_true",
        default=False,
        help="Save the full decoded output text for each request in requests.jsonl.",
    )
    parser.add_argument(
        "--server-log",
        default=None,
        help="Path to SGLang server log file. If set, the log slice for this run is copied into the run directory.",
    )
    parser.add_argument(
        "--instant-accept-chat",
        action="store_true",
        default=False,
        help="Record that instant-accept-chat was enabled (metadata only, actual policy is server-side).",
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
        record_output=args.record_output,
        server_log=args.server_log,
        instant_accept_chat=args.instant_accept_chat,
    )
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()