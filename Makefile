.PHONY: install test lint smoke full

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	pytest

lint:
	ruff check src tests scripts

smoke:
	python scripts/run_toy_experiment.py --config configs/toy_smoke.yaml

full:
	python scripts/run_toy_experiment.py --config configs/toy_full.yaml
