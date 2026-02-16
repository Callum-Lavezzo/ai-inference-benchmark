#!/usr/bin/env python3
"""Run a single local inference with mlx-lm."""

from __future__ import annotations

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local generation with mlx-lm.")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="HF model repo compatible with mlx-lm.",
    )
    parser.add_argument(
        "--prompt",
        default="Write one sentence about Apple Silicon efficiency.",
        help="Prompt text.",
    )
    parser.add_argument("--max-tokens", type=int, default=32, help="Max new tokens.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from mlx_lm import generate, load
    except Exception as exc:
        print("Failed to import mlx_lm. Did you install requirements?", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}")
    start_load = time.perf_counter()
    model, tokenizer = load(args.model)
    load_s = time.perf_counter() - start_load

    print(f"Generating (max_tokens={args.max_tokens}, temperature={args.temperature})...")
    start_gen = time.perf_counter()
    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temperature,
        verbose=False,
    )
    gen_s = time.perf_counter() - start_gen

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Output ===")
    print(text)
    print("\n=== Timing ===")
    print(f"load_seconds: {load_s:.3f}")
    print(f"generation_seconds: {gen_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
