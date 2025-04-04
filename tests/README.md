# Tests for PyTau

This directory contains tests for the PyTau package.

## Running Tests

Tests can be run using pytest:

```bash
pytest
```

Or using tox to test across multiple Python environments:

```bash
tox
```

## Test Structure

- `test_changepoint_model.py`: Tests for the changepoint model classes
- `conftest.py`: Pytest configuration and fixtures

## Adding Tests

When adding new models to the codebase, please add corresponding tests to ensure functionality.
