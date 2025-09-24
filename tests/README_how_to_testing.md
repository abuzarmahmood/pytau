# How-To Testing Documentation

This document describes the testing setup for PyTau's how-to notebooks and scripts.

## Overview

The testing suite ensures that all how-to materials (notebooks, scripts, and examples) are functional and can be executed without errors. This helps maintain the quality of documentation and examples provided to users.

## Test Files

### `test_how_to_notebooks.py`
Tests all Jupyter notebooks in the how-to directories:
- `pytau/how_to/notebooks/` - Main walkthrough notebooks
- `pytau/how_to/examples/` - Example notebooks

**Test Coverage:**
- Notebook execution using nbmake
- Verification that expected notebooks exist
- Directory structure validation

**Notebooks Tested:**
- `PyTau_Walkthrough_no_handlers.ipynb`
- `PyTau_Walkthrough_with_handlers.ipynb`
- `Bayesian_Changepoint_Model.ipynb`

### `test_how_to_scripts.py`
Tests all Python and shell scripts in the how-to directories:
- `pytau/how_to/scripts/` - Utility scripts
- `pytau/how_to/examples/` - Example scripts

**Test Coverage:**
- Python syntax validation
- Shell script basic validation
- Import structure checking (where applicable)
- Verification that expected scripts exist

**Scripts Tested:**
- `fit_manually.py`
- `fit_with_fit_handler.py`
- `changepoint_fit_test.py`
- `database_handler_test.py`
- `download_test_data.sh`

## Running Tests

### Local Testing

Run all how-to tests:
```bash
pytest tests/test_how_to_notebooks.py tests/test_how_to_scripts.py -v
```

Run notebook execution tests:
```bash
pytest --nbmake pytau/how_to/notebooks/ pytau/how_to/examples/ -v
```

Run with tox (includes all how-to testing):
```bash
tox
```

### CI/CD Integration

The tests are integrated into:
- **GitHub Actions**: Runs on all PRs and pushes
- **Tox configuration**: Includes nbmake testing for all how-to directories
- **Coverage reporting**: Ensures comprehensive testing

## Test Configuration

### Dependencies
The following packages are required for testing:
- `pytest` - Main testing framework
- `nbmake` - Notebook execution testing
- `jupyter` - Notebook support
- `nbconvert` - Notebook conversion utilities

### Markers
Tests use standard pytest markers:
- Parametrized tests for multiple files
- Skip markers for scripts requiring specific setup

## Maintenance

### Adding New How-To Materials
When adding new notebooks or scripts:
1. Place them in the appropriate directory (`notebooks/`, `scripts/`, or `examples/`)
2. Update the expected files list in the corresponding test file
3. Ensure the new materials follow existing patterns
4. Run tests locally to verify functionality

### Troubleshooting
Common issues and solutions:
- **Import errors**: Check that all required dependencies are installed
- **Notebook execution failures**: Verify notebooks can run in a clean environment
- **Script syntax errors**: Use a linter to check Python syntax
- **Missing files**: Ensure all expected files are present in the repository

## Integration with Existing Tests

The how-to tests complement the existing test suite:
- `test_changepoint_*.py` - Core functionality tests
- `conftest.py` - Shared test fixtures
- Model notebook tests - Already integrated in tox.ini

This comprehensive testing ensures that users can rely on the how-to materials to learn and use PyTau effectively.