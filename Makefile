.PHONY: fmt test lint

clean:
	rm -fr .venv clean .pytest_cache .ruff_cache .coverage coverage.xml
	rm -fr **/*.pyc

.venv/bin/python:
	pip install hatch
	hatch env create
	hatch run pip install ".[all]"

dev: .venv/bin/python
	@hatch run which python

fmt:
	black src tests
	ruff check src tests --fix

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src tests
	black --check src tests
