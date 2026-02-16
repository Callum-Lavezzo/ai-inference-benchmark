# ai-benchmark-m1

Local LLM inference + benchmarking + plotting on Apple Silicon using `mlx` and `mlx-lm`.

## Project layout

```text
ai-benchmark-m1/
  README.md
  requirements.txt
  .gitignore
  src/
    run_model.py
    benchmark.py
    plot_results.py
  results/
    .gitkeep
```

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3)
- Python 3.10+

## Setup (venv workflow)

```bash
cd ai-benchmark-m1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick smoke tests (small prompt)

Use a small quantized model and low token counts to keep runs short.

```bash
# Module form
python -m src.run_model --help
python -m src.benchmark --help
python -m src.plot_results --help

# Script form
python src/run_model.py --help
python src/benchmark.py --help
python src/plot_results.py --help
```

Optional real inference smoke test (downloads model if not cached):

```bash
python -m src.run_model \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "Say hello in one sentence." \
  --max-tokens 24
```

## Benchmark example

```bash
python -m src.benchmark \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "Explain overfitting in one sentence." \
  --runs 3 \
  --max-tokens 32 \
  --output results/benchmark_latest.csv
```

## Plot example

```bash
python -m src.plot_results \
  --input results/benchmark_latest.csv \
  --output results/benchmark_latest.png
```

## Notes

- `mlx-lm` handles model loading and generation.
- First run can be slower due to model download/loading.
- Keep prompts and `--max-tokens` small for smoke checks.
