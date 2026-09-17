"""
Emission-model components for composable changepoint models.

An EmissionModel owns: (1) per-state emission parameter priors, (2) how
those parameters combine with a transition weight_stack into a
time-varying value, and (3) the observation likelihood.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as tt


class EmissionModel:
    """Base class for emission components."""

    def build_params(self, data_array, batch_shape):
        """Declare per-state emission parameter priors on the currently-open
        model context and return them (shape/type is emission-specific)."""
        raise NotImplementedError

    def build_weight_stack(self, tau, batch_shape, n_timepoints):
        """Build the transition weight tensor this emission needs from
        tau. Defaults to the categorical-blend style used by most models."""
        from .transitions import blend_weights
        return blend_weights(tau, batch_shape, n_timepoints)

    def combine(self, params, weight_stack, batch_shape):
        """Combine per-state params with the weight_stack into a
        time-varying value (or tuple of values, e.g. (mu, sigma))."""
        raise NotImplementedError

    def likelihood(self, time_varying, data_array):
        """Declare the observation likelihood on the currently-open model
        context and return it."""
        raise NotImplementedError


class PoissonEmission(EmissionModel):
    """Poisson emission with no hierarchical structure. Supports the two
    batch shapes used by existing models: () for a single 1D time series
    (matches PoissonChangepoint1D) and (trials,) for per-trial data
    (matches SingleTastePoisson / SingleTastePoissonDirichlet).

    Args:
        n_states (int): number of states.
        combined_rate_name (str, optional): if given, the combined
            time-varying rate is wrapped in a named pm.Deterministic (some
            legacy models, e.g. the Dirichlet-process variants, register
            this as "lambda_" for inspection/plotting; others leave it
            anonymous). Defaults to None (anonymous, matching
            SingleTastePoisson/PoissonChangepoint1D).
    """

    def __init__(self, n_states, combined_rate_name=None):
        self.n_states = n_states
        self.combined_rate_name = combined_rate_name

    def build_params(self, data_array, batch_shape):
        n_states = self.n_states
        if len(batch_shape) == 0:
            # Unbatched: data_array is a flat 1D time series
            mean_vals = np.array(
                [np.mean(x) for x in np.array_split(data_array, n_states)])
            mean_vals += 0.01
            lambda_latent = pm.Exponential(
                "lambda", 1 / mean_vals, shape=n_states)
        elif len(batch_shape) == 1:
            # Per-trial: data_array is trials x neurons x time
            mean_vals = np.array(
                [np.mean(x, axis=-1)
                 for x in np.array_split(data_array, n_states, axis=-1)]
            ).T
            mean_vals = np.mean(mean_vals, axis=1)
            mean_vals += 0.01
            nrns = data_array.shape[1]
            lambda_latent = pm.Exponential(
                "lambda", 1 / mean_vals, shape=(nrns, n_states))
        else:
            raise NotImplementedError(
                f"PoissonEmission does not yet support batch_shape={batch_shape}")
        return lambda_latent

    def combine(self, lambda_latent, weight_stack, batch_shape):
        if len(batch_shape) == 0:
            # lambda_latent: (n_states,); weight_stack: (n_states, length)
            lambda_ = lambda_latent.dot(weight_stack)
        elif len(batch_shape) == 1:
            # weight_stack: (trials, n_states, length); lambda_latent: (nrns, n_states)
            lambda_ = tt.tensordot(
                weight_stack, lambda_latent, [1, 1]).swapaxes(1, 2)
        else:
            raise NotImplementedError(
                f"PoissonEmission does not yet support batch_shape={batch_shape}")
        if self.combined_rate_name is not None:
            lambda_ = pm.Deterministic(self.combined_rate_name, lambda_)
        return lambda_

    def likelihood(self, lambda_, data_array):
        return pm.Poisson("obs", lambda_, observed=data_array)


class NormalEmission(EmissionModel):
    """Unbatched Gaussian emission (matches GaussianChangepointMean2D,
    GaussianChangepointMeanVar2D, and GaussianChangepointMeanDirichlet,
    which differ only in the parameters below).

    Args:
        n_states (int): number of states.
        include_variance (bool): if True, sigma has its own per-state value
            blended through the weight_stack like mu (matches
            GaussianChangepointMeanVar2D); if False, a single sigma per
            data dimension is broadcast across all states/time (matches
            GaussianChangepointMean2D / GaussianChangepointMeanDirichlet).
            Defaults to False.
        mean_param_name (str): variable name for the per-state mean prior.
            Defaults to "mu"; GaussianChangepointMeanDirichlet uses
            "lambda" instead (a legacy naming inconsistency preserved for
            exact equivalence, not a Poisson rate despite the name).
        mu_prior_sigma (float): prior standard deviation for the per-state
            mean. Defaults to 1.0; GaussianChangepointMeanDirichlet uses
            10.0 (a more diffuse prior).
        sigma_prior_scale (float or array): HalfCauchy scale for sigma.
            Defaults to 3.0; GaussianChangepointMeanDirichlet uses a
            data-derived per-dimension std instead of a fixed constant.
        combined_mean_name (str, optional): if given, the combined
            time-varying mean is wrapped in a named pm.Deterministic
            (GaussianChangepointMeanDirichlet registers this as "lambda_").
            Defaults to None (anonymous).
    """

    def __init__(self, n_states, include_variance=False, mean_param_name="mu",
                 mu_prior_sigma=1.0, sigma_prior_scale=3.0,
                 combined_mean_name=None):
        self.n_states = n_states
        self.include_variance = include_variance
        self.mean_param_name = mean_param_name
        self.mu_prior_sigma = mu_prior_sigma
        self.sigma_prior_scale = sigma_prior_scale
        self.combined_mean_name = combined_mean_name

    def build_params(self, data_array, batch_shape):
        """
        Args:
            data_array (2D Numpy array): dimensions x time.
            batch_shape (tuple): expected to be ().
        """
        n_states = self.n_states
        mean_vals = np.array(
            [np.mean(x, axis=-1)
             for x in np.array_split(data_array, n_states, axis=-1)]
        ).T
        mean_vals += 0.01

        y_dim = data_array.shape[0]
        mu = pm.Normal(
            self.mean_param_name, mu=mean_vals, sigma=self.mu_prior_sigma,
            shape=(y_dim, n_states))
        sigma_shape = (y_dim, n_states) if self.include_variance else (y_dim,)
        sigma = pm.HalfCauchy(
            "sigma", self.sigma_prior_scale, shape=sigma_shape)
        return mu, sigma

    def combine(self, params, weight_stack, batch_shape):
        # weight_stack: (n_states, length); mu/sigma: (y_dim, n_states) or (y_dim,)
        mu, sigma = params
        mu_latent = mu.dot(weight_stack)
        if self.combined_mean_name is not None:
            mu_latent = pm.Deterministic(self.combined_mean_name, mu_latent)
        if self.include_variance:
            sigma_latent = sigma.dot(weight_stack)
        else:
            sigma_latent = sigma.dimshuffle(0, "x")
        return mu_latent, sigma_latent

    def likelihood(self, time_varying, data_array):
        mu_latent, sigma_latent = time_varying
        return pm.Normal(
            "obs", mu=mu_latent, sigma=sigma_latent, observed=data_array)
