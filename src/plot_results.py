#!/usr/bin/env python3
"""Plot benchmark CSV output to PNG."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark CSV results.")
    parser.add_argument(
        "--input",
        default="results/benchmark_latest.csv",
        help="Input CSV path produced by benchmark.py",
    )
    parser.add_argument(
        "--output",
        default="results/benchmark_latest.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="MLX-LM Benchmark",
        help="Plot title.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Input CSV not found: {in_path}", file=sys.stderr)
        return 2

    runs = []
    latencies = []
    tok_s = []

    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                runs.append(int(row["run"]))
                latencies.append(float(row["latency_seconds"]))
                tok_s.append(float(row["estimated_tokens_per_second"]))
            except Exception:
                continue

    if not runs:
        print("No plottable rows found in input CSV.", file=sys.stderr)
        return 3

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Failed to import matplotlib. Did you install requirements?", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(runs, latencies, marker="o", label="Latency (s)", color="#1f77b4")
    line2 = ax2.plot(
        runs,
        tok_s,
        marker="s",
        label="Estimated tokens/s",
        color="#ff7f0e",
    )

    ax1.set_xlabel("Run")
    ax1.set_ylabel("Latency (seconds)", color="#1f77b4")
    ax2.set_ylabel("Estimated tokens/second", color="#ff7f0e")
    ax1.set_title(args.title)
    ax1.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Wrote plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
