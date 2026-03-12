.PHONY: fmt test lint

fmt:
	black src tests
	ruff check src tests --fix

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src tests
	black --check src tests
