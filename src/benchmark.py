#!/usr/bin/env python3
"""Benchmark mlx-lm generation and write CSV results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import subprocess
import sys
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of falling back to synthetic benchmark data.",
    )
    return parser.parse_args()


def ensure_results_path(path_str: str) -> Path:
    path = Path(path_str)
    results_root = Path("results")
    if not path.is_absolute() and "results" not in path.parts:
        path = results_root / path
    return path


def run_worker(args: argparse.Namespace) -> tuple[float, list[dict]] | None:
    cmd = [
        sys.executable,
        "-m",
        "src.mlx_worker",
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--runs",
        str(args.runs),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
        return float(payload["load_seconds"]), list(payload["rows"])
    except Exception:
        return None


def synthetic_rows(args: argparse.Namespace) -> tuple[float, list[dict]]:
    # Fallback keeps the benchmark pipeline usable where MLX cannot initialize.
    load_seconds = 0.0
    rows = []
    for run_idx in range(1, args.runs + 1):
        latency = 0.06 + (run_idx * 0.01)
        new_tokens = args.max_tokens
        tok_s = (new_tokens / latency) if latency > 0 else 0.0
        rows.append(
            {
                "run": run_idx,
                "latency_seconds": latency,
                "estimated_new_tokens": new_tokens,
                "estimated_tokens_per_second": tok_s,
            }
        )
    return load_seconds, rows


def main() -> int:
    args = parse_args()

    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 2

    out_path = ensure_results_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    print(f"Running {args.runs} benchmark iterations...")

    worker_result = run_worker(args)
    mode = "real"
    if worker_result is None:
        if args.strict:
            print(
                "mlx-lm worker failed (possibly no available Metal device). "
                "Re-run without --strict to allow fallback.",
                file=sys.stderr,
            )
            return 1
        print(
            "mlx-lm worker failed; using synthetic fallback benchmark data.",
            file=sys.stderr,
        )
        load_seconds, raw_rows = synthetic_rows(args)
        mode = "synthetic"
    else:
        load_seconds, raw_rows = worker_result

    rows = []
    for raw in raw_rows:
        run_idx = int(raw["run"])
        elapsed = float(raw["latency_seconds"])
        new_tokens = int(raw["estimated_new_tokens"])
        tok_s = float(raw["estimated_tokens_per_second"])
        rows.append(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "run": run_idx,
                "mode": mode,
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
            f"run={run_idx} mode={mode} latency={elapsed:.3f}s "
            f"estimated_new_tokens={new_tokens} tok/s={tok_s:.2f}"
        )

    fieldnames = [
        "timestamp",
        "run",
        "mode",
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
    print(f"mode: {mode}")
    print(f"load_seconds: {load_seconds:.3f}")
    print(f"latency_avg_seconds: {statistics.mean(latencies):.3f}")
    print(f"latency_min_seconds: {min(latencies):.3f}")
    print(f"latency_max_seconds: {max(latencies):.3f}")
    print(f"tokens_per_second_avg: {statistics.mean(throughputs):.2f}")
    print(f"wrote_csv: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
