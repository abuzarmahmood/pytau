"""
Tests for composable changepoint models.
"""

import numpy as np
import pymc as pm
import pytest

from pytau.changepoint_model import gen_test_array
from pytau.composable import (
    ComposableModel,
    DirichletChangepoint,
    FixedChangepoint,
    GaussianEmission,
    GaussianLikelihood,
    PoissonEmission,
    PoissonLikelihood,
    create_gaussian_fixed_sigmoid,
    create_model,
    create_poisson_fixed_sigmoid,
    list_available_models,
)


class TestChangepointComponents:
    """Test changepoint components."""

    def test_fixed_changepoint_sigmoid(self):
        """Test fixed changepoint with sigmoid transitions."""
        n_states = 3
        time_length = 100

        changepoint = FixedChangepoint(n_states, change_type='sigmoid')

        with pm.Model():
            tau, weight_stack = changepoint.generate_changepoints(time_length)

        assert weight_stack.eval().shape == (n_states, time_length)

    def test_fixed_changepoint_step(self):
        """Test fixed changepoint with step transitions."""
        n_states = 3
        time_length = 100

        changepoint = FixedChangepoint(n_states, change_type='step')

        with pm.Model():
            tau, weight_stack = changepoint.generate_changepoints(time_length)

        assert weight_stack.eval().shape == (n_states, time_length)

    def test_dirichlet_changepoint(self):
        """Test Dirichlet process changepoint."""
        max_states = 10
        time_length = 100

        changepoint = DirichletChangepoint(max_states, change_type='sigmoid')

        with pm.Model():
            tau, weight_stack = changepoint.generate_changepoints(time_length)

        assert weight_stack.eval().shape == (max_states, time_length)


class TestEmissionComponents:
    """Test emission components."""

    def test_poisson_emission_2d(self):
        """Test Poisson emission for 2D data."""
        data = gen_test_array((10, 100), n_states=3, type='poisson')
        n_states = 3

        emission = PoissonEmission(data.shape, n_states)

        with pm.Model():
            lambda_vals = emission.generate_emissions(data)

        assert lambda_vals.eval().shape == (data.shape[0], n_states)

    def test_poisson_emission_3d(self):
        """Test Poisson emission for 3D data."""
        data = gen_test_array((5, 10, 100), n_states=3, type='poisson')
        n_states = 3

        emission = PoissonEmission(data.shape, n_states)

        with pm.Model():
            lambda_vals = emission.generate_emissions(data)

        assert lambda_vals.eval().shape == (data.shape[1], n_states)

    def test_gaussian_emission_mean_only(self):
        """Test Gaussian emission with mean only."""
        data = gen_test_array((10, 100), n_states=3, type='normal')
        n_states = 3

        emission = GaussianEmission(
            data.shape, n_states, include_variance=False)

        with pm.Model():
            mu, sigma = emission.generate_emissions(data)

        assert mu.eval().shape == (data.shape[0], n_states)
        assert sigma.eval().shape == (data.shape[0],)

    def test_gaussian_emission_mean_var(self):
        """Test Gaussian emission with mean and variance."""
        data = gen_test_array((10, 100), n_states=3, type='normal')
        n_states = 3

        emission = GaussianEmission(
            data.shape, n_states, include_variance=True)

        with pm.Model():
            mu, sigma = emission.generate_emissions(data)

        assert mu.eval().shape == (data.shape[0], n_states)
        assert sigma.eval().shape == (data.shape[0], n_states)


class TestLikelihoodComponents:
    """Test likelihood components."""

    def test_poisson_likelihood_2d(self):
        """Test Poisson likelihood for 2D data."""
        data = gen_test_array((10, 100), n_states=3, type='poisson')
        n_states = 3

        with pm.Model():
            # Create mock emissions and weights
            lambda_vals = pm.Exponential(
                "lambda", 1.0, shape=(data.shape[0], n_states))
            weight_stack = pm.math.sigmoid(
                np.random.randn(n_states, data.shape[1]))

            likelihood = PoissonLikelihood()
            obs = likelihood.generate_likelihood(
                data, lambda_vals, weight_stack)

        assert obs.name == "obs"

    def test_gaussian_likelihood_2d(self):
        """Test Gaussian likelihood for 2D data."""
        data = gen_test_array((10, 100), n_states=3, type='normal')
        n_states = 3

        with pm.Model():
            # Create mock emissions and weights
            mu = pm.Normal("mu", 0, 1, shape=(data.shape[0], n_states))
            sigma = pm.HalfCauchy("sigma", 1.0, shape=(data.shape[0],))
            weight_stack = pm.math.sigmoid(
                np.random.randn(n_states, data.shape[1]))

            likelihood = GaussianLikelihood()
            obs = likelihood.generate_likelihood(
                data, (mu, sigma), weight_stack)

        assert obs.name == "obs"


class TestComposableModel:
    """Test the full composable model."""

    def test_poisson_fixed_sigmoid_model(self):
        """Test Poisson model with fixed sigmoid changepoints."""
        data = gen_test_array((5, 10, 100), n_states=3, type='poisson')
        n_states = 3

        changepoint_comp = FixedChangepoint(n_states, change_type='sigmoid')
        emission_comp = PoissonEmission(data.shape, n_states)
        likelihood_comp = PoissonLikelihood()

        model_instance = ComposableModel(
            data, changepoint_comp, emission_comp, likelihood_comp)

        model = model_instance.generate_model()

        # Test that model can be compiled
        with model:
            # Just check that we can create the model
            assert len(model.basic_RVs) > 0

    def test_gaussian_fixed_sigmoid_model(self):
        """Test Gaussian model with fixed sigmoid changepoints."""
        data = gen_test_array((10, 100), n_states=3, type='normal')
        n_states = 3

        changepoint_comp = FixedChangepoint(n_states, change_type='sigmoid')
        emission_comp = GaussianEmission(
            data.shape, n_states, include_variance=False)
        likelihood_comp = GaussianLikelihood()

        model_instance = ComposableModel(
            data, changepoint_comp, emission_comp, likelihood_comp)

        model = model_instance.generate_model()

        # Test that model can be compiled
        with model:
            # Just check that we can create the model
            assert len(model.basic_RVs) > 0

    def test_model_test_method(self):
        """Test the model's test method."""
        data = gen_test_array((5, 10, 100), n_states=3, type='poisson')

        model_instance = create_poisson_fixed_sigmoid(data, n_states=3)

        # This should run without errors
        result = model_instance.test()
        assert result is True


class TestExampleFunctions:
    """Test example convenience functions."""

    def test_create_poisson_fixed_sigmoid(self):
        """Test convenience function for Poisson fixed sigmoid model."""
        data = gen_test_array((5, 10, 100), n_states=3, type='poisson')

        model_instance = create_poisson_fixed_sigmoid(data, n_states=3)
        model = model_instance.generate_model()

        with model:
            assert len(model.basic_RVs) > 0

    def test_create_gaussian_fixed_sigmoid(self):
        """Test convenience function for Gaussian fixed sigmoid model."""
        data = gen_test_array((10, 100), n_states=3, type='normal')

        model_instance = create_gaussian_fixed_sigmoid(data, n_states=3)
        model = model_instance.generate_model()

        with model:
            assert len(model.basic_RVs) > 0

    def test_create_model_function(self):
        """Test the generic create_model function."""
        data = gen_test_array((10, 100), n_states=3, type='poisson')

        model_instance = create_model(
            'poisson_fixed_sigmoid', data, n_states=3)
        model = model_instance.generate_model()

        with model:
            assert len(model.basic_RVs) > 0

    def test_create_model_invalid_type(self):
        """Test create_model with invalid type."""
        data = gen_test_array((10, 100), n_states=3, type='poisson')

        with pytest.raises(ValueError, match="Unknown model_type"):
            create_model('invalid_type', data, n_states=3)

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert 'poisson_fixed_sigmoid' in models
        assert 'gaussian_fixed_sigmoid' in models


if __name__ == "__main__":
    # Run a simple test if executed directly
    print("Running basic composable model test...")

    # Test Poisson model
    data = gen_test_array((5, 10, 100), n_states=3, type='poisson')
    model_instance = create_poisson_fixed_sigmoid(data, n_states=3)
    model = model_instance.generate_model()
    print("✅ Poisson model created successfully")

    # Test Gaussian model
    data = gen_test_array((10, 100), n_states=3, type='normal')
    model_instance = create_gaussian_fixed_sigmoid(data, n_states=3)
    model = model_instance.generate_model()
    print("✅ Gaussian model created successfully")

    print("All basic tests passed!")
