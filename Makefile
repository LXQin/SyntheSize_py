# Makefile for synthesize Python Package

.PHONY: help install install-dev clean test coverage lint format docs docs-serve build publish docs-clean publish-test check init-dev version info

# Default target
help:
	@echo "SyntheSize - Makefile Commands"
	@echo "================================"
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install package in regular mode"
	@echo "  make install-dev   - Install package with development dependencies"
	@echo "  make init-dev      - Initialize development environment (virtualenv)"
	@echo "  make clean         - Remove build artifacts and cache files"
	@echo "  make check         - Run linting and tests"
	@echo "  make info          - Show Python and package info"
	@echo "  make version       - Show package version"
	@echo ""
	@echo "Code Quality:"
	@echo "  make test          - Run tests with pytest"
	@echo "  make coverage      - Run tests with coverage report"
	@echo "  make lint          - Run linting (ruff)"
	@echo "  make format        - Format code with ruff"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Build documentation"
	@echo "  make docs-serve    - Build and serve documentation"
	@echo "  make docs-clean    - Clean documentation build"
	@echo ""
	@echo "Distribution:"
	@echo "  make build         - Build distribution packages"
	@echo "  make publish       - Publish to PyPI (requires credentials)"
	@echo "  make publish-test  - Publish to TestPyPI"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v

coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=synthesize --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Code Quality
lint:
	@echo "Running ruff linting..."
	ruff check synthesize/ tests/ --statistics
	ruff check synthesize/ tests/ --exit-zero --max-complexity=10 --line-length=100 --statistics

format:
	@echo "Formatting code with ruff..."
	ruff format synthesize/ tests/

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built in docs/_build/html/"

docs-serve: docs
	@echo "Serving documentation at http://localhost:8000"
	python -m http.server 8000 -d docs/_build/html/

docs-clean:
	@echo "Cleaning documentation build..."
	cd docs && make clean

# Distribution
build: clean
	@echo "Building distribution packages..."
	python -m build
	@echo "Build complete! Packages in dist/"

publish: build
	@printf "Are you sure you want to publish to PyPI? Type 'yes' to continue (default: no): "; \
	read -r resp; \
	resp=$$(echo "$$resp" | tr '[:upper:]' '[:lower:]'); \
	if [ "$$resp" != "yes" ]; then \
		echo "Publish aborted."; \
		exit 1; \
	fi; \
	echo "Publishing to PyPI..."; \
	python -m twine upload dist/*

publish-test: build
	@printf "Are you sure you want to publish to TestPyPI? Type 'yes' to continue (default: no): "; \
	read -r resp; \
	resp=$$(echo "$$resp" | tr '[:upper:]' '[:lower:]'); \
	if [ "$$resp" != "yes" ]; then \
		echo "Publish aborted."; \
		exit 1; \
	fi; \
	echo "Publishing to TestPyPI..."; \
	python -m twine upload --repository testpypi dist/*

# Development helpers
check: lint test
	@echo "All checks passed!"

init-dev:  ## Initialize development environment
	@echo "Initializing development environment..."
	python -m venv .venv
	.venv/bin/pip install --upgrade pip setuptools wheel
	.venv/bin/pip install -e ".[dev,docs]"
	@echo ""
	@echo "Development environment created. Activate with:"
	@echo "  source .venv/bin/activate  # On Unix/MacOS"
	@echo "  .venv\\Scripts\\activate  # On Windows"

# Version management
version:
	@grep "version" pyproject.toml | head -1

# Show Python and package info
info:
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "Package version:"
	@grep "version" pyproject.toml | head -1
	@echo ""
	@echo "Installed packages:"
	@pip list | grep -E "(synthesize|torch|pandas|numpy|scikit-learn)"
