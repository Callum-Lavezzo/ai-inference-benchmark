#!/usr/bin/env python3
"""Isolated mlx-lm benchmark worker.

This module runs in a subprocess so native MLX crashes do not take down
the parent benchmark command.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mlx-lm benchmark worker.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
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

    from mlx_lm import generate, load

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    load_seconds = time.perf_counter() - t0

    rows = []
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
                "run": run_idx,
                "latency_seconds": elapsed,
                "estimated_new_tokens": new_tokens,
                "estimated_tokens_per_second": tok_s,
            }
        )

    payload = {
        "load_seconds": load_seconds,
        "latency_avg_seconds": statistics.mean(r["latency_seconds"] for r in rows),
        "rows": rows,
    }
    print(json.dumps(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
