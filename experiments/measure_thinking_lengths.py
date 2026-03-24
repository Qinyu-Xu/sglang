"""
Measure output token length distribution for the two request classes.

  Chat class    — 100 ShareGPT prompts, enable_thinking=False
  Reasoning class — 100 AIME math problems, enable_thinking=True

All requests are sent concurrently. Results are plotted as two histograms
so you can see the real output-length gap between the classes.

Usage:
    python measure_thinking_lengths.py
    python measure_thinking_lengths.py --endpoint http://localhost:30000
    python measure_thinking_lengths.py --n-chat 100 --n-reasoning 100 \\
        --output data/results/thinking_lengths.png \\
        --results-json data/results/thinking_lengths.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="http://127.0.0.1:30000")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--n-chat", type=int, default=50,
                   help="Number of ShareGPT chat prompts")
    p.add_argument("--n-reasoning", type=int, default=50,
                   help="Number of AIME reasoning problems")
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--output", default="thinking_length_distribution.png")
    p.add_argument("--results-json", default="")
    return p.parse_args()





# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_chat_prompts(n: int) -> list[str]:
    """First n human turns from ShareGPT (no thinking)."""
    from datasets import load_dataset

    candidates = [
        ("Aeala/ShareGPT_Vicuna_unfiltered", "train"),
        ("anon8231489123/ShareGPT_Vicuna_unfiltered", "train"),
    ]
    for repo, split in candidates:
        try:
            print(f"Loading chat dataset ({repo}) …")
            ds = load_dataset(repo, split=split, streaming=True)
            first = next(iter(ds))
            if not first.get("conversations"):
                raise ValueError("no 'conversations' field")
            break
        except Exception as e:
            print(f"  {repo}: {e}")
            ds = None
    else:
        print("ERROR: could not load ShareGPT dataset.")
        sys.exit(1)

    prompts: list[str] = []
    for item in ds:
        convs = item.get("conversations") or []
        if not convs:
            continue
        text = convs[0].get("value", "").strip()
        if text:
            prompts.append(text)
        if len(prompts) >= n:
            break

    print(f"  → {len(prompts)} chat prompts")
    return prompts


def load_reasoning_prompts(n: int) -> list[str]:
    """AIME problems (with thinking)."""
    from datasets import load_dataset

    candidates = [
        ("AI-MO/aimo-validation-aime", "train"),
        ("AI-MO/aimo-validation-math", "train"),
    ]
    for repo, split in candidates:
        try:
            print(f"Loading reasoning dataset ({repo}) …")
            ds = load_dataset(repo, split=split, streaming=True)
            first = next(iter(ds))
            field = next((k for k in ("problem", "question", "Problem") if k in first), None)
            if not field:
                raise ValueError(f"no problem field in keys {list(first.keys())}")
            break
        except Exception as e:
            print(f"  {repo}: {e}")
            ds = None
            field = None
    else:
        print("ERROR: could not load AIME dataset.")
        sys.exit(1)

    prompts: list[str] = []
    for item in ds:
        text = item.get(field, "").strip()
        if text:
            prompts.append(text)
        if len(prompts) >= n:
            break

    if len(prompts) < n:
        print(f"  WARNING: only {len(prompts)} AIME problems available (requested {n}).")
    print(f"  → {len(prompts)} reasoning prompts")
    return prompts


# ---------------------------------------------------------------------------
# Single streaming request — returns (output_tokens, thinking_tokens)
# ---------------------------------------------------------------------------

def _call(
    endpoint: str,
    model: str,
    prompt: str,
    enable_thinking: bool,
    max_new_tokens: int,
) -> tuple[int, int]:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )

    output_tokens = 0
    thinking_tokens = 0

    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Count from delta chunks only — SGLang's usage.completion_tokens
            # lumps thinking + output together with no separate reasoning_tokens field.
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            think_piece = delta.get("reasoning_content") or delta.get("reasoning")
            token_piece = delta.get("content")
            if think_piece:
                thinking_tokens += 1
            if token_piece:
                output_tokens += 1

    return output_tokens, thinking_tokens


# ---------------------------------------------------------------------------
# Concurrent measurement
# ---------------------------------------------------------------------------

def run_measurements(
    chat_prompts: list[str],
    reasoning_prompts: list[str],
    endpoint: str,
    model: str,
    max_new_tokens: int,
) -> dict[str, list[int]]:
    """
    Fires all requests concurrently:
      - chat prompts      → enable_thinking=False
      - reasoning prompts → enable_thinking=True

    Returns dict with keys:
      chat_out, reasoning_out, reasoning_think
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # tag each task: ("chat", i) or ("reasoning", i)
    tasks = (
        [("chat",      i) for i in range(len(chat_prompts))] +
        [("reasoning", i) for i in range(len(reasoning_prompts))]
    )
    total = len(tasks)

    def _submit(kind: str, i: int):
        prompt = chat_prompts[i] if kind == "chat" else reasoning_prompts[i]
        thinking = (kind == "reasoning")
        out, thnk = _call(endpoint, model, prompt, thinking, max_new_tokens)
        return kind, i, out, thnk

    chat_out:        list[int] = [0] * len(chat_prompts)
    reasoning_out:   list[int] = [0] * len(reasoning_prompts)
    reasoning_think: list[int] = [0] * len(reasoning_prompts)
    errors: set[tuple[str, int]] = set()
    done = 0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=total) as pool:
        futures = {pool.submit(_submit, kind, i): (kind, i) for kind, i in tasks}
        for fut in as_completed(futures):
            done += 1
            kind, i = futures[fut]
            try:
                _, _, out, thnk = fut.result()
                if kind == "chat":
                    chat_out[i] = out
                    short = chat_prompts[i][:60].replace("\n", " ")
                    print(f"[{done:>3}/{total}] chat      {i:>3}: {out:>5} out  '{short}…'",
                          flush=True)
                else:
                    reasoning_out[i]   = out
                    reasoning_think[i] = thnk
                    short = reasoning_prompts[i][:60].replace("\n", " ")
                    print(f"[{done:>3}/{total}] reasoning {i:>3}: {out:>5} out  "
                          f"({thnk:>5} think)  '{short}…'", flush=True)
            except Exception as e:
                errors.add((kind, i))
                print(f"[{done:>3}/{total}] {kind} {i}: ERROR {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\nAll requests finished in {elapsed:.1f}s  ({len(errors)} errors)")

    # Remove errored indices
    errored_chat      = {i for k, i in errors if k == "chat"}
    errored_reasoning = {i for k, i in errors if k == "reasoning"}
    chat_out        = [v for i, v in enumerate(chat_out)        if i not in errored_chat]
    reasoning_out   = [v for i, v in enumerate(reasoning_out)   if i not in errored_reasoning]
    reasoning_think = [v for i, v in enumerate(reasoning_think) if i not in errored_reasoning]

    return {"chat_out": chat_out, "reasoning_out": reasoning_out,
            "reasoning_think": reasoning_think}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histograms(results: dict[str, list[int]], output_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib/numpy not available — skipping plot.")
        return

    chat_out        = results["chat_out"]
    reasoning_out   = results["reasoning_out"]
    reasoning_think = results["reasoning_think"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Qwen3 output-length distribution  "
        f"(chat n={len(chat_out)}, reasoning n={len(reasoning_out)})",
        fontsize=13,
    )

    def _hist(ax, data, title, color):
        arr = np.array(data)
        bins = min(60, max(10, len(arr) // 3))
        ax.hist(arr, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(arr), color="black", linestyle="--", linewidth=1.2,
                   label=f"median {np.median(arr):.0f}")
        ax.axvline(np.mean(arr), color="red", linestyle=":", linewidth=1.2,
                   label=f"mean {np.mean(arr):.0f}")
        ax.set_title(title)
        ax.set_xlabel("tokens")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    _hist(axes[0], chat_out,        "Chat (ShareGPT, no thinking)",      "#4C72B0")
    _hist(axes[1], reasoning_out,   "Reasoning (AIME, visible output)",  "#DD8452")
    _hist(axes[2], reasoning_think, "Reasoning (AIME, <think> tokens)",  "#55A868")

    if chat_out and reasoning_out:
        ratio = np.median(reasoning_out) / max(np.median(chat_out), 1)
        fig.text(0.5, 0.01,
                 f"Median output ratio  reasoning / chat = {ratio:.1f}×",
                 ha="center", fontsize=11, color="darkred")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Histogram saved → {out.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("=== Loading datasets ===")
    chat_prompts      = load_chat_prompts(args.n_chat)
    reasoning_prompts = load_reasoning_prompts(args.n_reasoning)

    print(f"\nEndpoint  : {args.endpoint}")
    print(f"Model     : {args.model}")
    print(f"Chat      : {len(chat_prompts)} prompts  (enable_thinking=False)")
    print(f"Reasoning : {len(reasoning_prompts)} prompts  (enable_thinking=True)")
    print(f"max_tokens : {args.max_new_tokens}\n")

    results = run_measurements(
        chat_prompts, reasoning_prompts,
        endpoint=args.endpoint,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    try:
        import numpy as np
        c  = np.array(results["chat_out"])
        r  = np.array(results["reasoning_out"])
        rk = np.array(results["reasoning_think"])
        print(f"\n{'='*60}")
        print(f"Chat output      — median {np.median(c):.0f}  mean {np.mean(c):.0f}"
              f"  p95 {np.percentile(c,95):.0f}")
        print(f"Reasoning output — median {np.median(r):.0f}  mean {np.mean(r):.0f}"
              f"  p95 {np.percentile(r,95):.0f}")
        print(f"Reasoning <think>— median {np.median(rk):.0f}  mean {np.mean(rk):.0f}"
              f"  p95 {np.percentile(rk,95):.0f}")
        print(f"\nMedian output ratio (reasoning / chat): {np.median(r)/max(np.median(c),1):.1f}×")
    except ImportError:
        pass

    if args.results_json:
        out_json = Path(args.results_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2))
        print(f"Raw counts saved → {out_json.resolve()}")

    plot_histograms(results, args.output)


if __name__ == "__main__":
    main()
