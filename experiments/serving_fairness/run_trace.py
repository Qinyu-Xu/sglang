"""Replay a workload spec against an OpenAI-compatible endpoint and log request traces."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from logger import DEFAULT_RESULTS_DIR, ExperimentLogger


@dataclass(frozen=True)
class TraceRequest:
    """One request entry from a workload specification."""

    request_id: str
    request_order: int
    workload_name: str
    class_label: str
    prompt_text: str
    submit_time_offset: float
    max_new_tokens_per_request: int
    prompt_type: str
    target_output_len: int


def replay_trace(
    workload_spec_path: str | Path,
    endpoint: str = "http://127.0.0.1:30000",
    model_path: str = "NousResearch/Meta-Llama-3-8B-Instruct",
    concurrency: int = 4,
    base_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Replay workload requests against endpoint with controlled concurrency."""
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")

    workload_name, requests = _read_workload_spec(workload_spec_path)
    logger = ExperimentLogger.init_run(base_dir=base_dir)
    logger.write_config(
        {
            "workload_name": workload_name,
            "workload_spec_path": str(workload_spec_path),
            "endpoint": endpoint,
            "model_path": model_path,
            "concurrency": concurrency,
        }
    )

    requests_sorted = sorted(
        requests, key=lambda r: (r.submit_time_offset, r.request_order)
    )
    run_start_ts = time.time()
    log_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: list[Future[dict[str, Any]]] = []
        for req in requests_sorted:
            _sleep_until_offset(run_start_ts, req.submit_time_offset)
            futures.append(executor.submit(_send_one_request, req, endpoint, model_path))

        for future in as_completed(futures):
            record = future.result()
            with log_lock:
                logger.log_request(record)

    return logger.run_dir


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
        submit_time_offset = _as_float(raw.get("submit_time_offset"), name="submit_time_offset")
        max_new_tokens = _as_int(
            raw.get("max_new_tokens", raw.get("max_new_tokens_per_request", 64)),
            name="max_new_tokens",
        )
        target_output_len = _as_int(
            raw.get("target_output_len", max_new_tokens), name="target_output_len"
        )
        prompt_type = str(raw.get("prompt_type", _default_prompt_type(class_label)))

        requests.append(
            TraceRequest(
                request_id=request_id,
                request_order=request_order,
                workload_name=workload_name,
                class_label=class_label,
                prompt_text=prompt_text,
                submit_time_offset=submit_time_offset,
                max_new_tokens_per_request=max_new_tokens,
                prompt_type=prompt_type,
                target_output_len=target_output_len,
            )
        )
    return workload_name, requests


def _send_one_request(
    request: TraceRequest, endpoint: str, model_path: str
) -> dict[str, Any]:
    submit_ts = time.time()
    first_token_ts: float | None = None
    finish_ts: float | None = None
    output_text_chunks: list[str] = []
    status = "success"
    error_msg = ""

    payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": request.prompt_text}],
        "max_tokens": request.max_new_tokens_per_request,
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
        with urllib.request.urlopen(http_request, timeout=120) as response:
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
                delta = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                )
                token_piece = delta.get("content")
                if token_piece:
                    output_text_chunks.append(str(token_piece))
                    if first_token_ts is None:
                        first_token_ts = time.time()
            finish_ts = time.time()
    except urllib.error.HTTPError as exc:
        status = "error"
        error_msg = f"HTTPError {exc.code}: {exc.reason}"
        finish_ts = time.time()
    except TimeoutError:
        status = "timeout"
        error_msg = "request timed out"
        finish_ts = time.time()
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        status = "error"
        error_msg = str(exc)
        finish_ts = time.time()

    output_len = _rough_token_count("".join(output_text_chunks))
    return {
        "request_id": request.request_id,
        "request_order": request.request_order,
        "workload_name": request.workload_name,
        "class_label": request.class_label,
        "submit_ts": submit_ts,
        "first_token_ts": first_token_ts,
        "finish_ts": finish_ts,
        "status": status,
        "prompt_len": _rough_token_count(request.prompt_text),
        "output_len": output_len,
        "error_msg": error_msg,
        "target_output_len": request.target_output_len,
        "max_new_tokens_per_request": request.max_new_tokens_per_request,
        "prompt_type": request.prompt_type,
    }


def _sleep_until_offset(run_start_ts: float, submit_time_offset: float) -> None:
    target_ts = run_start_ts + submit_time_offset
    while True:
        now = time.time()
        remaining = target_ts - now
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))


def _rough_token_count(text: str) -> int:
    return len(text.split())


def _default_prompt_type(class_label: str) -> str:
    return "reasoning" if "reason" in class_label else "chat"


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


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay workload trace against SGLang endpoint.")
    parser.add_argument("--workload-spec", required=True, help="Path to workload JSON file.")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:30000",
        help="OpenAI-compatible endpoint base URL.",
    )
    parser.add_argument(
        "--model-path",
        default="NousResearch/Meta-Llama-3-8B-Instruct",
        help="Model id passed in OpenAI requests.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of in-flight requests.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Base directory for run_YYYYMMDD_HHMMSS output folders.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    run_dir = replay_trace(
        workload_spec_path=args.workload_spec,
        endpoint=args.endpoint,
        model_path=args.model_path,
        concurrency=args.concurrency,
        base_dir=args.base_dir,
    )
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
