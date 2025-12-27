# Multi-Agent Investment Analysis System - Makefile
# Convenient commands for development and deployment

.PHONY: help install test lint format clean docker-build docker-run run-quick run-deep

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
POETRY := poetry
DOCKER := docker
TICKER ?= AAPL

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Multi-Agent Investment Analysis System$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make install          # Install dependencies"
	@echo "  make run-quick TICKER=AAPL  # Quick analysis for AAPL"
	@echo "  make test             # Run tests"
	@echo "  make docker-run TICKER=NVDA # Run with Docker"

install: ## Install dependencies using Poetry
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(POETRY) install
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

install-dev: ## Install dependencies including dev tools
	@echo "$(BLUE)Installing dependencies with dev tools...$(NC)"
	$(POETRY) install --with dev
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(POETRY) update
	@echo "$(GREEN)Dependencies updated successfully!$(NC)"

run: ## Run analysis (default: AAPL)
	@echo "$(BLUE)Running analysis for $(TICKER)...$(NC)"
	$(POETRY) run python -m src.main --ticker $(TICKER)

run-quick: ## Run quick analysis (default: AAPL)
	@echo "$(BLUE)Running quick analysis for $(TICKER)...$(NC)"
	$(POETRY) run python -m src.main --ticker $(TICKER) --quick

run-deep: ## Run deep analysis (default: AAPL)
	@echo "$(BLUE)Running deep analysis for $(TICKER)...$(NC)"
	$(POETRY) run python -m src.main --ticker $(TICKER)

run-verbose: ## Run with verbose logging (default: AAPL)
	@echo "$(BLUE)Running verbose analysis for $(TICKER)...$(NC)"
	$(POETRY) run python -m src.main --ticker $(TICKER) --verbose

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(POETRY) run pytest -v

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(POETRY) run pytest --cov=src --cov-report=html --cov-report=term-missing -v
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	$(POETRY) run pytest-watch

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(POETRY) run ruff check src/
	@echo "$(GREEN)Linting complete!$(NC)"

lint-fix: ## Run linting and auto-fix issues
	@echo "$(BLUE)Running linting with auto-fix...$(NC)"
	$(POETRY) run ruff check --fix src/
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with Black
	@echo "$(BLUE)Formatting code...$(NC)"
	$(POETRY) run black src/
	@echo "$(GREEN)Code formatted!$(NC)"

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(POETRY) run black --check src/

typecheck: ## Run type checking with MyPy
	@echo "$(BLUE)Running type checks...$(NC)"
	$(POETRY) run mypy src/
	@echo "$(GREEN)Type checking complete!$(NC)"

check-all: format-check lint typecheck ## Run all code quality checks
	@echo "$(GREEN)All checks passed!$(NC)"

clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-all: clean ## Clean everything including venv and ChromaDB
	@echo "$(BLUE)Cleaning everything...$(NC)"
	rm -rf .venv
	rm -rf chroma_db
	@echo "$(GREEN)Deep cleanup complete!$(NC)"

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER) build -t investment-agent:latest .
	@echo "$(GREEN)Docker image built successfully!$(NC)"

docker-run: ## Run with Docker (default: AAPL quick)
	@echo "$(BLUE)Running Docker container for $(TICKER)...$(NC)"
	$(DOCKER) run --rm --env-file .env investment-agent:latest --ticker $(TICKER) --quick

docker-run-deep: ## Run deep analysis with Docker (default: AAPL)
	@echo "$(BLUE)Running Docker container for $(TICKER) (deep mode)...$(NC)"
	$(DOCKER) run --rm --env-file .env investment-agent:latest --ticker $(TICKER)

docker-shell: ## Open shell in Docker container
	@echo "$(BLUE)Opening shell in Docker container...$(NC)"
	$(DOCKER) run --rm -it --env-file .env --entrypoint /bin/bash investment-agent:latest

docker-compose-up: ## Start services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"

docker-compose-down: ## Stop services with docker-compose
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)Services stopped!$(NC)"

env-setup: ## Copy .env.example to .env
	@if [ ! -f .env ]; then \
		echo "$(BLUE)Creating .env file from .env.example...$(NC)"; \
		cp .env.example .env; \
		echo "$(YELLOW)Please edit .env and add your API keys!$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists!$(NC)"; \
	fi

shell: ## Open Poetry shell
	$(POETRY) shell

deps-export: ## Export dependencies to requirements.txt
	@echo "$(BLUE)Exporting dependencies...$(NC)"
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	@echo "$(GREEN)Dependencies exported to requirements.txt$(NC)"

security-check: ## Run security vulnerability checks
	@echo "$(BLUE)Running security checks...$(NC)"
	$(POETRY) run safety check
	$(POETRY) run bandit -r src/
	@echo "$(GREEN)Security checks complete!$(NC)"

pre-commit: check-all test ## Run all pre-commit checks
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

ci: install check-all test ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"
