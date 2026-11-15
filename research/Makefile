.PHONY: install test quick run clean

install:
	pip install -r requirements.txt

test:
	python -m pytest -q tests

quick:
	python experiment.py --quick --out results_quick

run:
	python experiment.py --out results

clean:
	rm -rf results_quick .pytest_cache __pycache__ mgc_jax/__pycache__ tests/__pycache__
