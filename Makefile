# Makefile for Math Agent Project

.PHONY: help install install-dev test lint format check clean docs build publish

# Default target
help:
	@echo "Math Agent - Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      Install the package in development mode"
	@echo "  install-dev  Install development dependencies"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  check        Run all checks (lint, format, type)"
	@echo "  type-check   Run type checking with mypy"
	@echo ""
	@echo "Development Commands:"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  test         Run tests (placeholder)"
	@echo "  docs         Generate documentation (placeholder)"
	@echo ""
	@echo "Build Commands:"
	@echo "  build        Build the package"
	@echo "  publish      Publish to PyPI (placeholder)"
	@echo ""
	@echo "Agent Commands:"
	@echo "  solve        Solve a sample problem"
	@echo "  config       Show current configuration"
	@echo "  interactive  Start interactive mode"
	@echo ""
	@echo "Git Commands:"
	@echo "  pre-commit   Install pre-commit hooks"
	@echo "  pre-commit-run Run pre-commit on all files"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Code Quality
lint:
	@echo "Running flake8..."
	flake8 math_agent baselines --exclude=mathematics_dataset
	@echo "Running mypy..."
	mypy math_agent --exclude mathematics_dataset
	@echo "Linting complete!"

format:
	@echo "Running black..."
	black math_agent baselines --exclude mathematics_dataset
	@echo "Running isort..."
	isort math_agent baselines --skip mathematics_dataset
	@echo "Running autoflake..."
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive math_agent baselines --exclude mathematics_dataset
	@echo "Formatting complete!"

type-check:
	@echo "Running mypy type checking..."
	mypy math_agent --exclude mathematics_dataset

check: lint type-check
	@echo "All checks passed!"

# Development
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

test:
	@echo "Tests not implemented yet"
	# pytest tests/ -v

docs:
	@echo "Documentation generation not implemented yet"
	# sphinx-build -b html docs docs/_build

# Build
build: clean
	@echo "Building package..."
	python -m build

publish:
	@echo "Publishing not implemented yet"
	# twine upload dist/*

# Agent Commands
solve:
	@echo "Solving sample problem..."
	python -m math_agent.cli solve "What is 2 + 3 * 4?"

config:
	@echo "Current configuration:"
	python -m math_agent.cli config --show

interactive:
	@echo "Starting interactive mode..."
	python -m math_agent.cli interactive

# Git Commands
pre-commit:
	@echo "Installing pre-commit hooks..."
	pre-commit install

pre-commit-run:
	@echo "Running pre-commit on all files..."
	pre-commit run --all-files

# Environment setup
env-setup:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Activate with: source venv/bin/activate"
	@echo "Then run: make install-dev"

# Quick development setup
dev-setup: env-setup install-dev pre-commit
	@echo "Development environment setup complete!"
	@echo "Activate with: source venv/bin/activate"

# Run evaluation on sample data
eval-sample:
	@echo "Running evaluation on sample data..."
	python -m math_agent.cli evaluate --dataset data/math_qa/interpolate/arithmetic__add_or_sub.txt --max-problems 10 --verbose

# Show project structure
tree:
	@echo "Project structure:"
	tree -I 'mathematics_dataset|__pycache__|*.egg-info|.git|.pytest_cache|.mypy_cache|venv|env|.venv|build|dist|logs|results'

# Show Python environment info
info:
	@echo "Python Environment Information:"
	@echo "=============================="
	@echo "Python version: $(shell python --version)"
	@echo "Python path: $(shell which python)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Virtual environment: $(VIRTUAL_ENV)"
	@echo ""
	@echo "Package information:"
	@echo "==================="
	@python -c "import math_agent; print(f'Math Agent version: {math_agent.__version__}')" 2>/dev/null || echo "Math Agent not installed"
	@echo ""
	@echo "System information:"
	@echo "=================="
	@echo "OS: $(shell uname -s)"
	@echo "Architecture: $(shell uname -m)"
	@echo "Date: $(shell date)"
