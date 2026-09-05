.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python
PKG := src/eap
IMAGE ?= enterprise-agent-platform
TAG ?= dev

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install the package and development dependencies
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

.PHONY: lint
lint: ## Lint
	$(PY) -m ruff check src tests

.PHONY: format
format: ## Format in place
	$(PY) -m ruff format src tests
	$(PY) -m ruff check src tests --fix

.PHONY: format-check
format-check: ## Verify formatting without changing files
	$(PY) -m ruff format --check src tests

.PHONY: types
types: ## Type check in strict mode
	$(PY) -m mypy

.PHONY: test
test: ## Run the whole suite
	$(PY) -m pytest -q

.PHONY: test-unit
test-unit: ## Run everything except the MCP subprocess integration tests
	$(PY) -m pytest -q -m "not integration"

.PHONY: coverage
coverage: ## Run with a coverage report
	$(PY) -m pytest --cov=$(PKG) --cov-report=term-missing --cov-report=xml

.PHONY: eval
eval: ## Run the evaluation suite as a gate
	$(PY) -m scripts.run_evals

.PHONY: check
check: lint format-check types test ## Everything CI runs

.PHONY: run
run: ## Start the API with reload
	$(PY) -m uvicorn eap.api.app:app --reload --port 8000

.PHONY: sbom
sbom: ## Generate a CycloneDX SBOM
	$(PY) -m pip install --quiet cyclonedx-bom
	$(PY) -m cyclonedx_py environment --output-format json --outfile sbom.json

.PHONY: audit
audit: ## Scan dependencies for known vulnerabilities
	$(PY) -m pip install --quiet pip-audit
	$(PY) -m pip_audit --strict

.PHONY: docker-build
docker-build: ## Build the container image
	docker build -f deploy/docker/Dockerfile -t $(IMAGE):$(TAG) .

.PHONY: docker-run
docker-run: ## Run the image locally
	docker run --rm -p 8000:8000 --env-file .env $(IMAGE):$(TAG)

.PHONY: clean
clean: ## Remove build and test artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage sbom.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
