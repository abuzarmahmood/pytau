"""
PyTau: Powerful Changepoint Detection for Neural Data

PyTau provides Bayesian changepoint detection models specifically designed
for neural spike train data. It offers both a modern scikit-learn style API
and backward compatibility with the original HDF5-based workflow.

Main API Classes
----------------
ChangepointDetector : Scikit-learn style interface for changepoint detection
fit_changepoint_model : Convenience function for quick model fitting

Legacy API (for backward compatibility)
---------------------------------------
FitHandler : Original pipeline class for HDF5-based workflows
EphysData : Data loading utilities for HDF5 files

Examples
--------
>>> import numpy as np
>>> from pytau import ChangepointDetector, set_verbose
>>>
>>> # Enable verbose output to see detailed progress
>>> set_verbose(True)
>>>
>>> # Generate synthetic spike data (trials x neurons x time)
>>> spike_data = np.random.poisson(2, size=(20, 5, 100))
>>>
>>> # Fit changepoint model
>>> detector = ChangepointDetector(n_states=3)
>>> detector.fit(spike_data)
>>>
>>> # Get predictions
>>> predictions = detector.predict()
>>> changepoints = predictions['changepoints']
"""

# Import model classes for advanced usage
from . import changepoint_analysis, changepoint_model, changepoint_preprocess

# Import new scikit-learn style API (recommended)
from .changepoint_api import ChangepointDetector, fit_changepoint_model

# Import legacy API for backward compatibility
from .changepoint_io import FitHandler

# Import configuration functions
from .config import get_verbose, reset_config, set_verbose, verbose_context
from .utils import EphysData

# Version info
__version__ = "0.2.0"

# Define what gets imported with "from pytau import *"
__all__ = [
    # New API (recommended)
    'ChangepointDetector',
    'fit_changepoint_model',

    # Configuration
    'set_verbose',
    'get_verbose',
    'verbose_context',
    'reset_config',

    # Legacy API
    'FitHandler',
    'EphysData',

    # Modules for advanced usage
    'changepoint_model',
    'changepoint_preprocess',
    'changepoint_analysis',
]
