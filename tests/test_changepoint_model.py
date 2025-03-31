"""
Tests for the changepoint_model module.
"""

import pytest
import numpy as np
from pytau.changepoint_model import (
    ChangepointModel,
    GaussianChangepointMeanVar2D,
    GaussianChangepointMeanDirichlet,
    GaussianChangepointMean2D,
    SingleTastePoissonDirichlet,
    SingleTastePoisson,
    SingleTastePoissonVarsig,
    SingleTastePoissonVarsigFixed,
    AllTastePoisson,
    AllTastePoissonVarsigFixed,
    SingleTastePoissonTrialSwitch,
    AllTastePoissonTrialSwitch,
    gen_test_array
)

# Test the base model class
def test_base_model_class():
    """Test that the base model class raises NotImplementedError."""
    model = ChangepointModel()
    with pytest.raises(NotImplementedError):
        model.generate_model()
    with pytest.raises(NotImplementedError):
        model.test()

# Test the test data generation function
def test_gen_test_array():
    """Test the gen_test_array function."""
    # Test Poisson data
    array_size = (5, 10, 100)
    n_states = 3
    data = gen_test_array(array_size, n_states, type='poisson')
    assert data.shape == array_size
    
    # Test Normal data
    data = gen_test_array(array_size, n_states, type='normal')
    assert data.shape == array_size
    
    # Test with invalid type
    with pytest.raises(AssertionError):
        gen_test_array(array_size, n_states, type='invalid')
    
    # Test with too few time points
    with pytest.raises(AssertionError):
        gen_test_array((5, 10, 2), n_states, type='poisson')

# Test model initialization and basic functionality
@pytest.mark.parametrize("model_class,data_shape,n_states,extra_args", [
    (GaussianChangepointMeanVar2D, (10, 100), 3, {}),
    (GaussianChangepointMeanDirichlet, (10, 100), None, {"max_states": 5}),
    (GaussianChangepointMean2D, (10, 100), 3, {}),
    (SingleTastePoissonDirichlet, (5, 10, 100), None, {"max_states": 5}),
    (SingleTastePoisson, (5, 10, 100), 3, {}),
    (SingleTastePoissonVarsig, (5, 10, 100), 3, {}),
    (SingleTastePoissonVarsigFixed, (5, 10, 100), 3, {"inds_span": 1}),
    (AllTastePoisson, (2, 5, 10, 100), 3, {}),
    (AllTastePoissonVarsigFixed, (2, 5, 10, 100), 3, {"inds_span": 1}),
    (SingleTastePoissonTrialSwitch, (5, 10, 100), 3, {"switch_components": 2}),
    (AllTastePoissonTrialSwitch, (2, 5, 10, 100), 3, {"switch_components": 2}),
])
def test_model_initialization(model_class, data_shape, n_states, extra_args):
    """Test that models can be initialized and generate a model."""
    # Generate test data
    data_type = 'normal' if model_class in [GaussianChangepointMeanVar2D, GaussianChangepointMeanDirichlet, GaussianChangepointMean2D] else 'poisson'
    test_data = gen_test_array(data_shape, n_states=3 if n_states is None else n_states, type=data_type)
    
    # Initialize model
    if n_states is None:
        model_instance = model_class(test_data, **extra_args)
    else:
        model_instance = model_class(test_data, n_states, **extra_args)
    
    # Check that model can be generated
    model = model_instance.generate_model()
    assert model is not None

# Test the run_all_tests function (minimal test to avoid long runtime)
def test_run_all_tests_imports():
    """Test that run_all_tests can be imported."""
    from pytau.changepoint_model import run_all_tests
    assert callable(run_all_tests)

# Test the utility functions
def test_extract_inferred_values():
    """Test the extract_inferred_values function."""
    from pytau.changepoint_model import extract_inferred_values
    
    # Create a mock trace
    class MockTrace:
        def __init__(self):
            self.varnames = ['tau', 'lambda']
            self._data = {
                'tau': np.random.rand(10, 5, 2),
                'lambda': np.random.rand(10, 5, 3)
            }
        
        def __getitem__(self, key):
            return self._data[key]
    
    trace = MockTrace()
    result = extract_inferred_values(trace)
    
    assert 'tau_samples' in result
    assert 'lambda_stack' in result
    assert result['tau_samples'].shape == trace['tau'].shape
    assert result['lambda_stack'].shape == (5, 10, 3)  # Swapped axes
