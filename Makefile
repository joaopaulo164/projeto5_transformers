.PHONY: help install install-dev lint format format-check type-check test coverage smoke experiment clean

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  install        Install runtime dependencies"
	@echo "  install-dev    Install dev dependencies and editable package"
	@echo "  lint           Run Ruff linter"
	@echo "  format         Format code with Black and Ruff"
	@echo "  format-check   Check code formatting"
	@echo "  type-check     Run mypy static type checking"
	@echo "  test           Run unit and integration tests"
	@echo "  coverage       Run tests with coverage report"
	@echo "  smoke          Execute fast smoke test pipeline"
	@echo "  experiment     Execute default scientific experiment pipeline"
	@echo "  clean          Remove temporary artifacts and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

lint:
	ruff check src tests scripts

format:
	black src tests scripts
	ruff check --fix src tests scripts

format-check:
	black --check src tests scripts
	ruff check src tests scripts

type-check:
	mypy src

test:
	pytest tests/

coverage:
	pytest --cov=sentiment_transformer --cov-report=term-missing --cov-report=html:artifacts/coverage_html tests/

smoke:
	$(PYTHON) -m sentiment_transformer.cli run --config configs/smoke.yaml

experiment:
	$(PYTHON) -m sentiment_transformer.cli run --config configs/default.yaml

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage artifacts/coverage_html artifacts/coverage.xml build dist *.egg-info
