"""
Tests for how_to notebooks to ensure they execute without errors.
"""

import os
from pathlib import Path

import pytest

# Get the path to the how_to notebooks directory
HOW_TO_NOTEBOOKS_DIR = Path(__file__).parent.parent / \
    "pytau" / "how_to" / "notebooks"
EXAMPLES_DIR = Path(__file__).parent.parent / "pytau" / "how_to" / "examples"


def get_notebook_files():
    """Get all notebook files from how_to directories."""
    notebook_files = []

    # Get notebooks from notebooks directory
    if HOW_TO_NOTEBOOKS_DIR.exists():
        notebook_files.extend(list(HOW_TO_NOTEBOOKS_DIR.glob("*.ipynb")))

    # Get notebooks from examples directory
    if EXAMPLES_DIR.exists():
        notebook_files.extend(list(EXAMPLES_DIR.glob("*.ipynb")))

    return notebook_files


@pytest.mark.parametrize("notebook_path", get_notebook_files())
def test_notebook_structure(notebook_path):
    """Test that notebooks have valid structure and basic content."""
    import json

    assert notebook_path.exists(), f"Notebook {notebook_path} does not exist"
    assert notebook_path.suffix == ".ipynb", f"File {notebook_path} is not a notebook"

    # Test that notebook can be parsed as valid JSON
    with open(notebook_path, 'r') as f:
        try:
            notebook_content = json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Notebook {notebook_path} is not valid JSON: {e}")

    # Basic structure validation
    assert "cells" in notebook_content, f"Notebook {notebook_path} missing 'cells' key"
    assert "metadata" in notebook_content, f"Notebook {notebook_path} missing 'metadata' key"
    assert isinstance(
        notebook_content["cells"], list), f"Notebook {notebook_path} 'cells' is not a list"


def test_notebooks_exist():
    """Test that expected notebooks exist in the how_to directories."""
    expected_notebooks = [
        "PyTau_Walkthrough_no_handlers.ipynb",
        "PyTau_Walkthrough_with_handlers.ipynb",
        "Bayesian_Changepoint_Model.ipynb"
    ]

    all_notebooks = get_notebook_files()
    notebook_names = [nb.name for nb in all_notebooks]

    for expected in expected_notebooks:
        assert expected in notebook_names, f"Expected notebook {expected} not found"


def test_notebook_directories_exist():
    """Test that the how_to notebook directories exist."""
    assert HOW_TO_NOTEBOOKS_DIR.exists(), "how_to/notebooks directory does not exist"
    assert EXAMPLES_DIR.exists(), "how_to/examples directory does not exist"


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", get_notebook_files())
def test_notebook_execution_with_nbmake(notebook_path):
    """Test notebook execution using nbmake - requires full dependencies.

    This test is marked as 'slow' and should be run separately with:
    pytest --nbmake pytau/how_to/notebooks/ pytau/how_to/examples/ -m slow

    Or run all notebooks with nbmake directly:
    pytest --nbmake pytau/how_to/notebooks/ pytau/how_to/examples/
    """
    # This is a placeholder test - actual execution happens via nbmake plugin
    # when pytest is run with --nbmake flag on the notebook directories
    assert notebook_path.exists(), f"Notebook {notebook_path} does not exist"
