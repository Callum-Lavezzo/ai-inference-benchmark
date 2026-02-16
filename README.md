# ai-benchmark-m1

Benchmark local LLM inference on Apple Silicon with reproducible, lightweight workflows.

This repository demonstrates practical AI performance evaluation skills relevant to an Entry-Level AI Solutions Architect / Field Applications Engineer role: environment setup, repeatable benchmarking, clean artifact management, and result visualization.

## Overview

- Purpose: measure latency and estimated throughput for local `mlx-lm` generation workloads.
- Platform focus: macOS on Apple Silicon (M1/M2/M3).
- Design goal: fast feedback loops using small defaults so the project is safe to run on a laptop.

## Repo Structure

```text
ai-benchmark-m1/
  Makefile
  README.md
  LICENSE
  requirements.txt
  src/
    run_model.py        # Single inference run
    benchmark.py        # Multi-run benchmark -> CSV
    plot_results.py     # CSV -> PNG plot
  results/
    .gitkeep
    plots/
```

## Quickstart

### 1) Setup

```bash
make setup
source .venv/bin/activate
```

### 2) Smoke Check (No heavy model run)

```bash
make smoke
```

This validates CLI entry points and plotting pipeline with a tiny synthetic CSV.

### 3) Run a Lightweight Benchmark

```bash
make bench
```

### 4) Plot Benchmark Output

```bash
make plot
```

## Methodology

1. Load an `mlx-lm` compatible model.
2. Execute N repeated generations with fixed prompt and token settings.
3. Record per-run latency and estimated tokens/sec.
4. Save benchmark artifacts as CSV under `results/`.
5. Generate plots under `results/plots/` for quick trend inspection.

Current default model is intentionally small and quantized:
`mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

## Results (Placeholder)

Use this section for recruiter-facing evidence once you run local benchmarks.

| Date | Hardware | Model | Runs | Avg Latency (s) | Avg Tok/s | Artifacts |
|---|---|---|---:|---:|---:|---|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | `results/benchmark_latest.csv`, `results/plots/benchmark_latest.png` |

## Reproducibility Notes

- Pin Python dependencies via `requirements.txt`.
- Keep benchmark settings explicit (`--runs`, `--max-tokens`, `--temperature`).
- Commit only code and lightweight placeholders, not generated benchmark artifacts.

## Troubleshooting

- `ModuleNotFoundError: mlx_lm`
  - Run `make setup`, then ensure virtualenv is activated.
- First benchmark run is slow
  - Initial model download/loading is expected and included in reported `load_seconds`.
- Plot command fails due to missing CSV
  - Run `make bench` first or provide `--input` pointing to an existing file under `results/`.

## Next Steps

- Add multi-model comparison runs and aggregate dashboards.
- Add CPU/GPU memory telemetry for deeper performance profiling.
- Add CI checks for `make smoke` on pull requests.
- Export Markdown summary reports from benchmark CSVs.

## Direct CLI Usage

```bash
python -m src.run_model --help
python -m src.benchmark --help
python -m src.plot_results --help
```
