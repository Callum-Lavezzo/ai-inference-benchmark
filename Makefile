PYTHON ?= python3
VENV ?= .venv
VENV_PIP := $(VENV)/bin/pip

MODEL ?= mlx-community/Qwen2.5-0.5B-Instruct-4bit
PROMPT ?= Explain why quick benchmark smoke tests are useful.
RUNS ?= 3
MAX_TOKENS ?= 32
TEMPERATURE ?= 0.2

BENCH_OUT ?= results/benchmark_latest.csv
PLOT_OUT ?= results/plots/benchmark_latest.png

.PHONY: setup smoke bench plot clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

smoke:
	mkdir -p results/plots
	$(PYTHON) -m src.run_model --help
	$(PYTHON) -m src.benchmark --help
	$(PYTHON) -m src.plot_results --help
	printf 'run,latency_seconds,estimated_tokens_per_second\n1,0.10,100.0\n2,0.12,95.0\n' > results/benchmark_smoke.csv
	@if $(PYTHON) -c "import matplotlib" >/dev/null 2>&1; then \
		$(PYTHON) -m src.plot_results --input benchmark_smoke.csv --output benchmark_smoke.png --title "Smoke Plot"; \
	else \
		echo "matplotlib not installed; skipping plot render in smoke check"; \
	fi

bench:
	mkdir -p results
	$(PYTHON) -m src.benchmark \
		--model "$(MODEL)" \
		--prompt "$(PROMPT)" \
		--runs $(RUNS) \
		--max-tokens $(MAX_TOKENS) \
		--temperature $(TEMPERATURE) \
		--output "$(BENCH_OUT)"

plot:
	mkdir -p results/plots
	$(PYTHON) -m src.plot_results --input "$(BENCH_OUT)" --output "$(PLOT_OUT)"

clean:
	rm -f results/*.csv results/*.json
	rm -f results/plots/*.png
