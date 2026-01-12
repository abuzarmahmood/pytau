"""
Likelihood component implementations.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as tt

from .base import LikelihoodComponent


class PoissonLikelihood(LikelihoodComponent):
    """Poisson likelihood for spike count data."""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)

    def generate_likelihood(self, data_array, emissions, weight_stack):
        """Generate Poisson likelihood.

        Args:
            data_array: Observed spike count data
            emissions: Lambda parameters from emission component
            weight_stack: State weights over time from changepoint component

        Returns:
            PyMC observation variable
        """
        lambda_latent = emissions

        # Calculate time-varying rates
        if len(data_array.shape) == 3:  # trials x neurons x time
            # Expand lambda for trials dimension
            lambda_expanded = tt.expand_dims(
                lambda_latent, 0)  # 1 x neurons x states
            lambda_expanded = tt.broadcast_to(
                lambda_expanded,
                (data_array.shape[0], data_array.shape[1],
                 lambda_latent.shape[-1])
            )  # trials x neurons x states

            # Apply weights: (trials x neurons x states) @ (states x time)
            lambda_final = tt.batched_dot(lambda_expanded, weight_stack)

        elif len(data_array.shape) == 2:  # neurons x time or trials x time
            # Apply weights: (neurons x states) @ (states x time)
            lambda_final = lambda_latent.dot(weight_stack)

        else:  # 1D time series
            # Apply weights: (states,) @ (states x time)
            lambda_final = lambda_latent.dot(weight_stack)

        # Poisson observation
        observation = pm.Poisson("obs", mu=lambda_final, observed=data_array)

        return observation


class GaussianLikelihood(LikelihoodComponent):
    """Gaussian likelihood for continuous data."""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)

    def generate_likelihood(self, data_array, emissions, weight_stack):
        """Generate Gaussian likelihood.

        Args:
            data_array: Observed continuous data
            emissions: (mu, sigma) parameters from emission component
            weight_stack: State weights over time from changepoint component

        Returns:
            PyMC observation variable
        """
        if isinstance(emissions, tuple):
            mu, sigma = emissions
            model_variance = True
        else:
            mu = emissions
            sigma = None
            model_variance = False

        # Calculate time-varying means
        if len(data_array.shape) == 3:  # trials x neurons x time
            # Expand mu for trials dimension if needed
            if len(mu.shape) == 2:  # neurons x states
                mu_expanded = tt.expand_dims(mu, 0)  # 1 x neurons x states
                mu_expanded = tt.broadcast_to(
                    mu_expanded,
                    (data_array.shape[0], data_array.shape[1], mu.shape[-1])
                )  # trials x neurons x states
            else:
                mu_expanded = mu

            # Apply weights
            mu_final = tt.batched_dot(mu_expanded, weight_stack)

            if model_variance and len(sigma.shape) == 2:
                # Handle variance similarly
                sigma_expanded = tt.expand_dims(sigma, 0)
                sigma_expanded = tt.broadcast_to(
                    sigma_expanded,
                    (data_array.shape[0], data_array.shape[1], sigma.shape[-1])
                )
                sigma_final = tt.batched_dot(sigma_expanded, weight_stack)
            else:
                sigma_final = sigma

        elif len(data_array.shape) == 2:  # neurons x time
            # Apply weights: (neurons x states) @ (states x time)
            mu_final = mu.dot(weight_stack)

            if model_variance and len(sigma.shape) == 2:
                sigma_final = sigma.dot(weight_stack)
            else:
                sigma_final = sigma

        else:  # 1D time series
            # Apply weights: (states,) @ (states x time)
            mu_final = mu.dot(weight_stack)

            if model_variance:
                sigma_final = sigma.dot(weight_stack)
            else:
                sigma_final = sigma

        # Gaussian observation
        observation = pm.Normal(
            "obs", mu=mu_final, sigma=sigma_final, observed=data_array)

        return observation
