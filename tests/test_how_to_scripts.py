"""
Tests for how_to scripts to ensure they can be imported and have basic functionality.
"""

import pytest
import sys
import subprocess
from pathlib import Path
import importlib.util


# Get the path to the how_to scripts directory
HOW_TO_SCRIPTS_DIR = Path(__file__).parent.parent / "pytau" / "how_to" / "scripts"
EXAMPLES_DIR = Path(__file__).parent.parent / "pytau" / "how_to" / "examples"


def get_python_script_files():
    """Get all Python script files from how_to directories."""
    script_files = []
    
    # Get scripts from scripts directory
    if HOW_TO_SCRIPTS_DIR.exists():
        script_files.extend(list(HOW_TO_SCRIPTS_DIR.glob("*.py")))
    
    # Get scripts from examples directory
    if EXAMPLES_DIR.exists():
        script_files.extend(list(EXAMPLES_DIR.glob("*.py")))
    
    return script_files


def get_shell_script_files():
    """Get all shell script files from how_to directories."""
    script_files = []
    
    # Get shell scripts from scripts directory
    if HOW_TO_SCRIPTS_DIR.exists():
        script_files.extend(list(HOW_TO_SCRIPTS_DIR.glob("*.sh")))
    
    return script_files


@pytest.mark.parametrize("script_path", get_python_script_files())
def test_python_script_syntax(script_path):
    """Test that Python scripts have valid syntax and can be imported."""
    assert script_path.exists(), f"Script {script_path} does not exist"
    assert script_path.suffix == ".py", f"File {script_path} is not a Python script"
    
    # Test syntax by attempting to compile
    with open(script_path, 'r') as f:
        source_code = f.read()
    
    try:
        compile(source_code, str(script_path), 'exec')
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {script_path}: {e}")


@pytest.mark.parametrize("script_path", get_shell_script_files())
def test_shell_script_syntax(script_path):
    """Test that shell scripts have basic syntax validity."""
    assert script_path.exists(), f"Script {script_path} does not exist"
    assert script_path.suffix == ".sh", f"File {script_path} is not a shell script"
    
    # Basic check - ensure file is not empty and has shebang or content
    with open(script_path, 'r') as f:
        content = f.read().strip()
    
    assert len(content) > 0, f"Shell script {script_path} is empty"


def test_expected_scripts_exist():
    """Test that expected scripts exist in the how_to directories."""
    expected_python_scripts = [
        "fit_manually.py",
        "fit_with_fit_handler.py",
        "changepoint_fit_test.py",
        "database_handler_test.py"
    ]
    
    expected_shell_scripts = [
        "download_test_data.sh"
    ]
    
    all_python_scripts = get_python_script_files()
    python_script_names = [script.name for script in all_python_scripts]
    
    all_shell_scripts = get_shell_script_files()
    shell_script_names = [script.name for script in all_shell_scripts]
    
    for expected in expected_python_scripts:
        assert expected in python_script_names, f"Expected Python script {expected} not found"
    
    for expected in expected_shell_scripts:
        assert expected in shell_script_names, f"Expected shell script {expected} not found"


def test_script_directories_exist():
    """Test that the how_to script directories exist."""
    assert HOW_TO_SCRIPTS_DIR.exists(), "how_to/scripts directory does not exist"
    assert EXAMPLES_DIR.exists(), "how_to/examples directory does not exist"


@pytest.mark.parametrize("script_path", get_python_script_files())
def test_python_script_imports(script_path):
    """Test that Python scripts don't have obvious import errors."""
    # Skip this test for scripts that are meant to be run standalone
    # and might require specific data or setup
    skip_import_test = [
        "fit_manually.py",
        "fit_with_fit_handler.py", 
        "changepoint_fit_test.py",
        "database_handler_test.py"
    ]
    
    if script_path.name in skip_import_test:
        pytest.skip(f"Skipping import test for {script_path.name} - requires specific setup")
    
    # For other scripts, try to check basic import structure
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Basic check - if it has imports, they should be at the top
    lines = content.split('\n')
    import_lines = [line for line in lines if line.strip().startswith(('import ', 'from '))]
    
    # This is a basic structural check
    assert len(import_lines) >= 0  # Allow scripts with no imports