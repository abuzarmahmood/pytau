"""
Global configuration settings for PyTau.

This module provides centralized configuration management for PyTau,
including verbose logging settings and other global parameters.
"""

import os
from typing import Union


class PyTauConfig:
    """
    Global configuration class for PyTau settings.

    This class manages global settings that affect the behavior of PyTau
    across all modules, including verbosity levels and logging preferences.

    Attributes
    ----------
    verbose : bool
        Global verbosity setting. When True, enables informative print
        statements throughout PyTau operations.
    """

    def __init__(self):
        # Initialize with default settings
        self._verbose = self._get_default_verbose()

    def _get_default_verbose(self) -> bool:
        """
        Get default verbose setting from environment variable or default.

        Returns
        -------
        bool
            Default verbose setting
        """
        # Check environment variable first
        env_verbose = os.environ.get('PYTAU_VERBOSE', '').lower()
        if env_verbose in ('true', '1', 'yes', 'on'):
            return True
        elif env_verbose in ('false', '0', 'no', 'off'):
            return False
        else:
            # Default to False for quiet operation
            return False

    @property
    def verbose(self) -> bool:
        """Get current verbose setting."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: Union[bool, int, str]):
        """
        Set verbose setting.

        Parameters
        ----------
        value : bool, int, or str
            Verbose setting. Accepts:
            - bool: True/False
            - int: 0 (False) or non-zero (True)
            - str: 'true'/'false', '1'/'0', 'yes'/'no', 'on'/'off'
        """
        if isinstance(value, bool):
            self._verbose = value
        elif isinstance(value, int):
            self._verbose = bool(value)
        elif isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ('true', '1', 'yes', 'on'):
                self._verbose = True
            elif value_lower in ('false', '0', 'no', 'off'):
                self._verbose = False
            else:
                raise ValueError(f"Invalid verbose string value: {value}")
        else:
            raise TypeError(
                f"Verbose setting must be bool, int, or str, got {type(value)}")

    def set_verbose(self, value: Union[bool, int, str]) -> None:
        """
        Set verbose setting (alternative method).

        Parameters
        ----------
        value : bool, int, or str
            Verbose setting
        """
        self.verbose = value

    def get_verbose(self) -> bool:
        """
        Get current verbose setting (alternative method).

        Returns
        -------
        bool
            Current verbose setting
        """
        return self.verbose

    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values."""
        self._verbose = self._get_default_verbose()


# Global configuration instance
_config = PyTauConfig()


def set_verbose(value: Union[bool, int, str]) -> None:
    """
    Set global verbose setting for PyTau.

    When verbose is True, PyTau will print informative messages about:
    - Model initialization and configuration
    - Method activation and progress
    - Data preprocessing steps
    - Inference progress and completion

    Parameters
    ----------
    value : bool, int, or str
        Verbose setting. Accepts:
        - bool: True/False
        - int: 0 (False) or non-zero (True)
        - str: 'true'/'false', '1'/'0', 'yes'/'no', 'on'/'off'

    Examples
    --------
    >>> import pytau
    >>> pytau.set_verbose(True)
    >>> # Now all PyTau operations will print informative messages
    >>>
    >>> pytau.set_verbose('off')
    >>> # Disable verbose output
    """
    _config.set_verbose(value)


def get_verbose() -> bool:
    """
    Get current global verbose setting.

    Returns
    -------
    bool
        Current verbose setting

    Examples
    --------
    >>> import pytau
    >>> pytau.get_verbose()
    False
    >>> pytau.set_verbose(True)
    >>> pytau.get_verbose()
    True
    """
    return _config.get_verbose()


def verbose_print(*args, **kwargs) -> None:
    """
    Print function that respects global verbose setting.

    Only prints if global verbose setting is True. Otherwise, does nothing.

    Parameters
    ----------
    *args
        Arguments passed to print()
    **kwargs
        Keyword arguments passed to print()

    Examples
    --------
    >>> from pytau.config import verbose_print
    >>> verbose_print("This will only print if verbose=True")
    """
    if _config.get_verbose():
        print(*args, **kwargs)


def reset_config() -> None:
    """
    Reset PyTau configuration to default values.

    This resets all global settings including verbose mode back to their
    default values (determined by environment variables or built-in defaults).

    Examples
    --------
    >>> import pytau
    >>> pytau.set_verbose(True)
    >>> pytau.reset_config()
    >>> pytau.get_verbose()  # Back to default (False)
    False
    """
    _config.reset_to_defaults()


# Context manager for temporary verbose setting
class verbose_context:
    """
    Context manager for temporarily changing verbose setting.

    Parameters
    ----------
    verbose : bool, int, or str
        Temporary verbose setting

    Examples
    --------
    >>> import pytau
    >>> pytau.set_verbose(False)
    >>> with pytau.verbose_context(True):
    ...     # Verbose output enabled only within this block
    ...     detector = pytau.ChangepointDetector()
    ...     detector.fit(data)
    >>> # Verbose setting restored to previous value
    """

    def __init__(self, verbose: Union[bool, int, str]):
        self.new_verbose = verbose
        self.old_verbose = None

    def __enter__(self):
        self.old_verbose = get_verbose()
        set_verbose(self.new_verbose)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_verbose(self.old_verbose)
