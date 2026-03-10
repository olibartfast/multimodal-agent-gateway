#!/usr/bin/env bash
# scripts/run.sh - Self-bootstrapping development script
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

# Install/upgrade package in editable mode
pip install --quiet --upgrade pip
pip install --quiet -e "${PROJECT_ROOT}[all]"

# Run the CLI with all arguments
exec vlm-agent-gateway "$@"
