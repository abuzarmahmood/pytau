# Notebook Testing with nbmake

This document describes the notebook testing setup for PyTau using [nbmake](https://github.com/treebeardtech/nbmake).

## Overview

The project uses `nbmake` to automatically test Jupyter notebooks by executing them and verifying they complete without errors. This ensures that example notebooks and tutorials remain functional as the codebase evolves.

## Setup

### Dependencies

The notebook testing requires the following additional dependencies:

- `nbmake`: Pytest plugin for testing notebooks
- `ipykernel`: Jupyter kernel for executing notebooks
- `nbclient`: Client for executing notebooks
- `nbformat`: Notebook format handling

These are included in the `dev` optional dependencies in `pyproject.toml`.

### Installation

```bash
pip install -e .[dev]
```

## Running Notebook Tests

### Individual Notebooks

Test a single notebook:

```bash
pytest --nbmake path/to/notebook.ipynb
```

### All Working Notebooks

Test all working notebooks using tox:

```bash
tox -e notebooks
```

### Manual Testing

Test specific notebooks manually:

```bash
# Test model demos
pytest --nbmake pytau/how_to/model_demos/GaussianChangepointMean2D_demo.ipynb

# Test examples
pytest --nbmake pytau/how_to/examples/Bayesian_Changepoint_Model.ipynb
```

## Test Configuration

### Pytest Configuration

The notebook tests are configured in `pytest.ini`:

- Custom marker `notebook` for notebook-specific tests
- Verbose output enabled
- Short traceback format for cleaner output

### Tox Configuration

A dedicated `notebooks` environment is configured in `tox.ini` that:

- Installs all required dependencies
- Tests only working notebooks (skips problematic ones)
- Runs notebooks individually for better error isolation

## Currently Tested Notebooks

The following notebooks are currently tested:

### Model Demos (`pytau/how_to/model_demos/`)
- ✅ `CategoricalChangepoint2D_demo.ipynb`
- ✅ `GaussianChangepointMean2D_demo.ipynb`
- ✅ `GaussianChangepointMeanDirichlet_demo.ipynb`
- ✅ `AllTastePoisson_demo.ipynb`
- ✅ `SingleTastePoisson_demo.ipynb`
- ✅ `GaussianChangepointMeanVar2D_demo.ipynb`

### Examples (`pytau/how_to/examples/`)
- ✅ `Bayesian_Changepoint_Model.ipynb`

## Known Issues

The following notebooks are currently skipped due to API issues:

### Notebooks (`pytau/how_to/notebooks/`)
- ❌ `PyTau_New_API_Example.ipynb` - AttributeError: 'InferenceData' object has no attribute 'varnames'
- ❌ `PyTau_Verbose_Example.ipynb` - AttributeError: 'InferenceData' object has no attribute 'varnames'
- ❌ `PyTau_Walkthrough_no_handlers.ipynb` - Similar API issues
- ❌ `PyTau_Walkthrough_with_handlers.ipynb` - Similar API issues

These notebooks need to be updated to work with the current PyMC/ArviZ API.

## CI/CD Integration

Notebook testing is integrated into the GitHub Actions workflow:

- Tests run on all supported Python versions (3.10, 3.11)
- Tests run on multiple operating systems (Ubuntu, macOS, Windows)
- Only working notebooks are tested to avoid CI failures
- Tests run after the main test suite

## Troubleshooting

### Common Issues

1. **Long execution times**: Some notebooks with MCMC sampling can take several minutes
2. **Memory usage**: Multiple notebooks running simultaneously may consume significant memory
3. **Platform differences**: Some notebooks may behave differently across operating systems

### Debugging Failed Notebooks

To debug a failing notebook:

1. Run the notebook manually in Jupyter
2. Check for missing dependencies
3. Verify data files are available
4. Check for API changes in dependencies

### Adding New Notebooks

To add a new notebook to testing:

1. Ensure the notebook runs successfully manually
2. Add it to the `tox.ini` notebooks environment
3. Consider adding it to the CI workflow if it's fast and stable
4. Update this documentation

## Performance Considerations

- Notebook tests are separated from unit tests for performance
- Individual notebook execution allows for better error isolation
- Timeout handling prevents hanging tests
- Parallel execution is avoided to prevent resource conflicts

## Future Improvements

- Add notebook execution timeout configuration
- Implement notebook output comparison for regression testing
- Add performance benchmarking for notebook execution times
- Consider using notebook parameterization for different test scenarios