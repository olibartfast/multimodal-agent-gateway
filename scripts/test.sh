#!/usr/bin/env bash
# scripts/test.sh - Self-bootstrapping test runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Create venv if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# Install/upgrade package with dev dependencies
pip install --quiet --upgrade pip
pip install --quiet -e "${PROJECT_ROOT}[dev,video]"

# Run linting
echo "Running ruff..."
ruff check src/ tests/ --fix

# Run type checking
echo "Running mypy..."
mypy src/ || true

# Run tests with coverage
echo "Running pytest..."
pytest tests/ --cov=src/vlm_agent_gateway --cov-report=term-missing "$@"
