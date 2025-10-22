# Laplace

A Python project using UV for dependency management.

## Getting Started

This project uses [UV](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

- Python 3.8 or higher
- UV installed

### Installation

1. Install UV if you haven't already:
   ```bash
   pip install uv
   ```

2. Install project dependencies:
   ```bash
   uv sync
   ```

3. Install development dependencies:
   ```bash
   uv sync --extra dev
   ```

### Development

- Run tests:
  ```bash
  uv run pytest
  ```

- Format and lint code (using Ruff - much faster than black/flake8):
  ```bash
  uv run ruff check src tests          # Check for linting issues
  uv run ruff check --fix src tests   # Auto-fix linting issues
  uv run ruff format src tests        # Format code
  ```

- Type checking:
  ```bash
  uv run mypy src
  ```

- Run all checks at once:
  ```bash
  uv run ruff check --fix src tests && uv run ruff format src tests && uv run mypy src
  ```

## Project Structure

```
laplace/
├── src/           # Source code
├── tests/         # Test files
├── pyproject.toml # Project configuration
├── .python-version # Python version specification
└── README.md      # This file
```
