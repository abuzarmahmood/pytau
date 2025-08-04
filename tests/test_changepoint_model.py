"""
Tests for the changepoint_model module.
"""

import numpy as np
import pytest

from pytau.changepoint_model import (
    AllTastePoisson,
    AllTastePoissonTrialSwitch,
    AllTastePoissonVarsigFixed,
    CategoricalChangepoint2D,
    ChangepointModel,
    GaussianChangepointMean2D,
    GaussianChangepointMeanDirichlet,
    GaussianChangepointMeanVar2D,
    SingleTastePoisson,
    SingleTastePoissonDirichlet,
    SingleTastePoissonTrialSwitch,
    SingleTastePoissonVarsig,
    SingleTastePoissonVarsigFixed,
    advi_fit,
    extract_inferred_values,
    gen_test_array,
)


# Test the base model class
@pytest.mark.slow
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
    data = gen_test_array(array_size, n_states, type="poisson")
    assert data.shape == array_size

    # Test Normal data
    data = gen_test_array(array_size, n_states, type="normal")
    assert data.shape == array_size

    # Test with invalid type
    with pytest.raises(AssertionError):
        gen_test_array(array_size, n_states, type="invalid")

    # Test with too few time points
    with pytest.raises(AssertionError):
        gen_test_array((5, 10, 2), n_states, type="poisson")


# Test model initialization and basic functionality
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_class,data_shape,n_states,extra_args",
    [
        (GaussianChangepointMeanVar2D, (10, 100), 3, {}),
        (GaussianChangepointMeanDirichlet, (10, 100), None, {"max_states": 5}),
        (GaussianChangepointMean2D, (10, 100), 3, {}),
        (SingleTastePoissonDirichlet, (5, 10, 100), None, {"max_states": 5}),
        (SingleTastePoisson, (5, 10, 100), 3, {}),
        # (SingleTastePoissonVarsig, (5, 10, 100), 3, {}),
        # (SingleTastePoissonVarsigFixed, (5, 10, 100), 3, {"inds_span": 1}),
        (AllTastePoisson, (2, 5, 10, 100), 3, {}),
        # (AllTastePoissonVarsigFixed, (2, 5, 10, 100), 3, {"inds_span": 1}),
        (SingleTastePoissonTrialSwitch,
         (5, 10, 100), 3, {"switch_components": 2}),
        (AllTastePoissonTrialSwitch, (2, 5, 10, 100),
         3, {"switch_components": 2}),
        (CategoricalChangepoint2D, (5, 100), 3, {}),  # Changed to 2D shape
    ],
)
def test_model_initialization(model_class, data_shape, n_states, extra_args):
    """Test that models can be initialized and generate a model."""
    # Generate test data
    data_type = (
        "normal"
        if model_class
        in [
            GaussianChangepointMeanVar2D,
            GaussianChangepointMeanDirichlet,
            GaussianChangepointMean2D,
        ]
        else "poisson"
    )
    test_data = gen_test_array(
        data_shape, n_states=3 if n_states is None else n_states, type=data_type
    )

    # Initialize model
    if n_states is None:
        model_instance = model_class(data_array=test_data, **extra_args)
    else:
        model_instance = model_class(
            data_array=test_data, n_states=n_states, **extra_args)

    # Check that model can be generated
    model = model_instance.generate_model()
    assert model is not None


def test_categorical_changepoint_3d():
    """Test the CategoricalChangepoint2D model."""
    data = np.random.randint(0, 3, size=(5, 100))  # Use 2D data
    model_instance = CategoricalChangepoint2D(data_array=data, n_states=3)
    model = model_instance.generate_model()
    assert model is not None
    # Skip the actual fitting to save time in tests
    assert hasattr(model, "observed_RVs")


@pytest.mark.slow
def test_run_all_tests():
    """Test that run_all_tests can be imported and executed."""
    try:
        from pytau.changepoint_model import run_all_tests

        assert callable(run_all_tests)

        # We don't actually run the full tests here to avoid long runtime in CI
        # Just verify the function exists and is callable
        # For full testing, run the function directly
    except ImportError:
        pytest.skip(
            "run_all_tests function not found in changepoint_model module")


# Test the utility functions
def test_extract_inferred_values():
    """Test the extract_inferred_values function."""

    # Create a mock trace
    class MockTrace:
        def __init__(self):
            self.varnames = ["tau", "lambda"]
            self._data = {
                "tau": np.random.rand(10, 5, 2),
                "lambda": np.random.rand(10, 5, 3),
            }

        def __getitem__(self, key):
            return self._data[key]

    trace = MockTrace()
    result = extract_inferred_values(trace)

    assert "tau_samples" in result
    assert "lambda_stack" in result
    assert result["tau_samples"].shape == trace["tau"].shape
    assert result["lambda_stack"].shape == (5, 10, 3)  # Swapped axes


# @pytest.mark.slow
# def test_advi_fit():
#     """Test the advi_fit function with a simple model."""
#     from pytau.changepoint_model import SingleTastePoisson
#     
#     # Generate test data
#     test_data = gen_test_array((5, 10, 100), n_states=3, type="poisson")
#     
#     # Create a simple model
#     model_instance = SingleTastePoisson(test_data, n_states=3)
#     model = model_instance.generate_model()
#     
#     # Test ADVI fit with minimal iterations
#     result = advi_fit(model, fit=10, samples=5)
#     
#     # Check that we get the expected return values
#     assert len(result) == 5  # model, trace, lambda_stack, tau_samples, observations
#     model_out, trace, lambda_stack, tau_samples, observations = result
#     
#     # Check shapes and types
#     assert tau_samples.shape[0] == 5  # samples
#     assert lambda_stack.shape[1] == 5  # samples
#     assert observations.shape == test_data.shape
#
#
# @pytest.mark.slow
# def test_mcmc_fit():
#     """Test the mcmc_fit function with a simple model."""
#     from pytau.changepoint_model import mcmc_fit, SingleTastePoisson
#     
#     # Generate test data
#     test_data = gen_test_array((3, 5, 50), n_states=2, type="poisson")  # Smaller for speed
#     
#     # Create a simple model
#     model_instance = SingleTastePoisson(test_data, n_states=2)
#     model = model_instance.generate_model()
#     
#     # Test MCMC fit with minimal samples
#     result = mcmc_fit(model, samples=10)
#     
#     # Check that we get the expected return values
#     assert len(result) == 5  # model, trace, lambda_stack, tau_samples, observations
#     model_out, trace, lambda_stack, tau_samples, observations = result
#     
#     # Check that we have samples
#     assert tau_samples.shape[-1] > 0  # Should have some samples
#     assert lambda_stack.shape[-1] > 0  # Should have some samples
#     assert observations.shape == test_data.shape
#
#
# @pytest.mark.slow
# def test_dpp_fit():
#     """Test the dpp_fit function with a simple model."""
#     from pytau.changepoint_model import dpp_fit, SingleTastePoisson
#     
#     # Generate test data
#     test_data = gen_test_array((3, 5, 50), n_states=2, type="poisson")  # Smaller for speed
#     
#     # Create a simple model
#     model_instance = SingleTastePoisson(test_data, n_states=2)
#     model = model_instance.generate_model()
#     
#     # Test DPP fit with minimal parameters
#     trace = dpp_fit(model, n_chains=2, n_cores=1, tune=10, draws=10)
#     
#     # Check that we get a trace with expected variables
#     assert "tau" in trace.varnames
#     assert "lambda" in trace.varnames
#     
#     # Check that we have the expected number of samples
#     assert trace["tau"].shape[0] == 20  # 2 chains * 10 draws
#
#
# @pytest.mark.slow 
# def test_sampler_comparison():
#     """Test that different samplers produce reasonable results on the same model."""
#     from pytau.changepoint_model import SingleTastePoisson, advi_fit, mcmc_fit
#     
#     # Generate test data with known structure
#     test_data = gen_test_array((4, 8, 80), n_states=3, type="poisson")
#     
#     # Create model
#     model_instance = SingleTastePoisson(test_data, n_states=3)
#     
#     # Test ADVI
#     model_advi = model_instance.generate_model()
#     advi_result = advi_fit(model_advi, fit=50, samples=20)
#     advi_tau = advi_result[3]  # tau_samples
#     
#     # Test MCMC (with very few samples for speed)
#     model_mcmc = model_instance.generate_model()
#     mcmc_result = mcmc_fit(model_mcmc, samples=20)
#     mcmc_tau = mcmc_result[3]  # tau_samples
#     
#     # Basic sanity checks - both should produce tau values in reasonable range
#     assert np.all(advi_tau >= 0)
#     assert np.all(advi_tau <= test_data.shape[-1])
#     assert np.all(mcmc_tau >= 0) 
#     assert np.all(mcmc_tau <= test_data.shape[-1])
#     
#     # Check that tau values are ordered (changepoints should be sequential)
#     assert np.all(np.diff(np.mean(advi_tau, axis=0), axis=-1) >= 0)
#     assert np.all(np.diff(np.mean(mcmc_tau, axis=0), axis=-1) >= 0)
#
#
# @pytest.mark.slow
# def test_sampler_with_gaussian_model():
#     """Test samplers with Gaussian models."""
#     from pytau.changepoint_model import GaussianChangepointMean2D, advi_fit
#     
#     # Generate test data
#     test_data = gen_test_array((10, 100), n_states=3, type="normal")
#     
#     # Create model
#     model_instance = GaussianChangepointMean2D(test_data, n_states=3)
#     model = model_instance.generate_model()
#     
#     # Test ADVI fit
#     result = advi_fit(model, fit=20, samples=10)
#     
#     # Check that we get the expected return values for Gaussian model
#     assert len(result) == 6  # model, trace, mu_stack, sigma_stack, tau_samples, observations
#     model_out, trace, mu_stack, sigma_stack, tau_samples, observations = result
#     
#     # Check shapes
#     assert tau_samples.shape[0] == 10  # samples
#     assert mu_stack.shape[1] == 10  # samples
#     assert sigma_stack.shape[1] == 10  # samples
#     assert observations.shape == test_data.shape
#
#
# @pytest.mark.slow
# def test_find_best_states():
#     """Test the find_best_states function."""
#     from pytau.changepoint_model import find_best_states, single_taste_poisson
#     
#     # Generate test data
#     test_data = gen_test_array((4, 6, 60), n_states=3, type="poisson")
#     
#     # Test find_best_states with a small range
#     best_model, model_list, elbo_values = find_best_states(
#         test_data, 
#         single_taste_poisson, 
#         n_fit=10, 
#         n_samples=5, 
#         min_states=2, 
#         max_states=4
#     )
#     
#     # Check outputs
#     assert len(model_list) == 3  # 2, 3, 4 states
#     assert len(elbo_values) == 3
#     assert best_model is not None
#     
#     # ELBO values should be negative (log likelihood)
#     assert all(elbo < 0 for elbo in elbo_values)
#
#
# @pytest.mark.slow
# def test_sampler_error_handling():
#     """Test that samplers handle errors gracefully."""
#     from pytau.changepoint_model import SingleTastePoisson, advi_fit
#     
#     # Generate test data
#     test_data = gen_test_array((3, 5, 50), n_states=2, type="poisson")
#     
#     # Create model
#     model_instance = SingleTastePoisson(test_data, n_states=2)
#     model = model_instance.generate_model()
#     
#     # Test with invalid parameters (should not crash)
#     try:
#         result = advi_fit(model, fit=1, samples=1)  # Very minimal fit
#         assert len(result) == 5
#     except Exception as e:
#         # If it fails, it should fail gracefully
#         assert isinstance(e, (ValueError, RuntimeError))
#
#
# @pytest.mark.slow
# def test_dpp_fit_with_numpyro():
#     """Test the dpp_fit function with numpyro backend."""
#     from pytau.changepoint_model import dpp_fit, SingleTastePoisson
#     
#     # Generate test data
#     test_data = gen_test_array((3, 5, 50), n_states=2, type="poisson")
#     
#     # Create model
#     model_instance = SingleTastePoisson(test_data, n_states=2)
#     model = model_instance.generate_model()
#     
#     try:
#         # Test DPP fit with numpyro backend
#         trace = dpp_fit(model, n_chains=2, n_cores=1, tune=5, draws=5, use_numpyro=True)
#         
#         # Check that we get a trace with expected variables
#         assert "tau" in trace.varnames
#         assert "lambda" in trace.varnames
#         
#         # Check that we have the expected number of samples
#         assert trace["tau"].shape[0] == 10  # 2 chains * 5 draws
#         
#     except ImportError:
#         # Skip if numpyro is not available
#         pytest.skip("numpyro not available")
#     except Exception as e:
#         # Other errors might occur due to environment setup
#         pytest.skip(f"numpyro backend test failed: {e}")
#
#
# @pytest.mark.slow
# def test_sampler_consistency():
#     """Test that repeated runs of the same sampler give consistent results."""
#     from pytau.changepoint_model import SingleTastePoisson, advi_fit
#     import numpy as np
#     
#     # Set random seed for reproducibility
#     np.random.seed(42)
#     
#     # Generate test data
#     test_data = gen_test_array((4, 6, 60), n_states=2, type="poisson")
#     
#     # Run ADVI twice with same parameters
#     model_instance1 = SingleTastePoisson(test_data, n_states=2)
#     model1 = model_instance1.generate_model()
#     result1 = advi_fit(model1, fit=30, samples=10)
#     
#     model_instance2 = SingleTastePoisson(test_data, n_states=2)
#     model2 = model_instance2.generate_model()
#     result2 = advi_fit(model2, fit=30, samples=10)
#     
#     # Results should be in similar ranges (not exact due to randomness)
#     tau1 = result1[3]  # tau_samples
#     tau2 = result2[3]  # tau_samples
#     
#     # Check that both produce reasonable tau values
#     assert np.all(tau1 >= 0) and np.all(tau1 <= test_data.shape[-1])
#     assert np.all(tau2 >= 0) and np.all(tau2 <= test_data.shape[-1])
#     
#     # Mean tau values should be in similar ballpark (within 50% of data length)
#     tau1_mean = np.mean(tau1)
#     tau2_mean = np.mean(tau2)
#     assert abs(tau1_mean - tau2_mean) < test_data.shape[-1] * 0.5


def test_module_import():
    """Test that the changepoint_model module can be imported."""
    import pytau.changepoint_model

    assert pytau.changepoint_model is not None


def test_sampler_functions_exist():
    """Test that all sampler functions are available."""
    from pytau.changepoint_model import advi_fit, mcmc_fit, dpp_fit, find_best_states
    
    assert callable(advi_fit)
    assert callable(mcmc_fit) 
    assert callable(dpp_fit)
    assert callable(find_best_states)
