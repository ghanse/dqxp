.PHONY: fmt test lint

clean:
	rm -fr .venv clean .pytest_cache .ruff_cache .coverage coverage.xml
	rm -fr **/*.pyc

.venv/bin/python:
	pip install hatch
	hatch env create

dev: .venv/bin/python
	@hatch run which python

fmt:
	hatch run black src tests
	hatch run ruff check src tests --fix

test:
	hatch run pytest tests/ -v --tb=short

lint:
	hatch run ruff check src tests
	hatch run black --check src tests
