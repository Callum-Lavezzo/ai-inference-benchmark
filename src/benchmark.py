#!/usr/bin/env python3
"""Benchmark mlx-lm generation and write CSV results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import statistics
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mlx-lm local generation.")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="HF model repo compatible with mlx-lm.",
    )
    parser.add_argument(
        "--prompt",
        default="Summarize why small smoke tests are useful.",
        help="Prompt text for each run.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument(
        "--output",
        default="results/benchmark_latest.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def estimate_new_tokens(tokenizer, prompt: str, generated: str) -> int:
    try:
        prompt_ids = tokenizer.encode(prompt)
        gen_ids = tokenizer.encode(generated)
        return max(0, len(gen_ids) - len(prompt_ids))
    except Exception:
        return 0


def main() -> int:
    args = parse_args()

    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_lm import generate, load
    except Exception as exc:
        print("Failed to import mlx_lm. Did you install requirements?", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    load_seconds = time.perf_counter() - t0

    rows = []
    print(f"Running {args.runs} benchmark iterations...")
    for run_idx in range(1, args.runs + 1):
        start = time.perf_counter()
        generated = generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temp=args.temperature,
            verbose=False,
        )
        elapsed = time.perf_counter() - start
        new_tokens = estimate_new_tokens(tokenizer, args.prompt, generated)
        tok_s = (new_tokens / elapsed) if elapsed > 0 else 0.0
        rows.append(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "run": run_idx,
                "model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "latency_seconds": f"{elapsed:.6f}",
                "estimated_new_tokens": new_tokens,
                "estimated_tokens_per_second": f"{tok_s:.6f}",
                "load_seconds": f"{load_seconds:.6f}",
            }
        )
        print(
            f"run={run_idx} latency={elapsed:.3f}s "
            f"estimated_new_tokens={new_tokens} tok/s={tok_s:.2f}"
        )

    fieldnames = [
        "timestamp",
        "run",
        "model",
        "max_tokens",
        "temperature",
        "latency_seconds",
        "estimated_new_tokens",
        "estimated_tokens_per_second",
        "load_seconds",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    latencies = [float(r["latency_seconds"]) for r in rows]
    throughputs = [float(r["estimated_tokens_per_second"]) for r in rows]
    print("\n=== Summary ===")
    print(f"runs: {args.runs}")
    print(f"load_seconds: {load_seconds:.3f}")
    print(f"latency_avg_seconds: {statistics.mean(latencies):.3f}")
    print(f"latency_min_seconds: {min(latencies):.3f}")
    print(f"latency_max_seconds: {max(latencies):.3f}")
    print(f"tokens_per_second_avg: {statistics.mean(throughputs):.2f}")
    print(f"wrote_csv: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
