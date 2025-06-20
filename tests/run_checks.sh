#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Go to the project root directory (one level up from tests/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Running checks from: $(pwd)"

echo "--- Running mypy for type checking ---"
mypy .

echo "--- Running black to format code ---"
black .

echo "--- Running isort to sort imports ---"
isort .

echo "--- Running pytest to run tests ---"
pytest

echo "--- All checks passed successfully! ---"
