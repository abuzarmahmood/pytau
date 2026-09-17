"""
Changepoint-prior components for composable changepoint models.

A ChangepointPrior produces `tau` (changepoint positions) only -- it does
not build the transition weight stack, since different emission models
need different weight tensors from the same tau (see transitions.py).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as tt

_HYPERPRIOR_FAMILIES = {
    "halfcauchy": pm.HalfCauchy, "halfnormal": pm.HalfNormal}


class ChangepointPrior:
    """Base class for changepoint-position priors."""

    @property
    def n_output_states(self):
        """Number of states this prior produces (n_states)."""
        raise NotImplementedError

    def build_tau(self, batch_shape, n_timepoints):
        """Declare PyMC random variables for tau on the currently-open model
        context and return the `tau` tensor, shape
        batch_shape + (n_output_states - 1,).
        """
        raise NotImplementedError


class FixedCountChangepoint(ChangepointPrior):
    """A fixed number of changepoints, each with an independent Beta-derived
    latent position (the "sorted Beta" prior used by most existing models).
    """

    def __init__(self, n_states, hyperprior="halfcauchy", hyper_scale=3.0,
                 tau_latent_initval=None):
        """
        Args:
            n_states (int): number of states (n_states - 1 changepoints).
            hyperprior (str): {'halfcauchy', 'halfnormal'} -- distribution
                family for the a_tau/b_tau Beta hyperpriors. Existing models
                are inconsistent here (single-taste models use halfcauchy,
                all-taste models use halfnormal); preserved as a
                constructor option rather than normalized, to keep exact
                per-model equivalence.
            hyper_scale (float): scale parameter for the hyperprior.
            tau_latent_initval (array-like, optional): explicit initval for
                the tau_latent Beta RV (some legacy models set this, others
                don't -- preserved per-model for exact equivalence).
        """
        if hyperprior not in _HYPERPRIOR_FAMILIES:
            raise ValueError(
                f"hyperprior must be one of {list(_HYPERPRIOR_FAMILIES)}")
        self.n_states = n_states
        self.hyperprior = hyperprior
        self.hyper_scale = hyper_scale
        self.tau_latent_initval = tau_latent_initval

    @property
    def n_output_states(self):
        return self.n_states

    def build_tau(self, batch_shape, n_timepoints):
        hyper = _HYPERPRIOR_FAMILIES[self.hyperprior]
        a_tau = hyper("a_tau", self.hyper_scale, shape=self.n_states - 1)
        b_tau = hyper("b_tau", self.hyper_scale, shape=self.n_states - 1)

        beta_kwargs = {}
        if self.tau_latent_initval is not None:
            beta_kwargs["initval"] = self.tau_latent_initval
        tau_latent = pm.Beta(
            "tau_latent", a_tau, b_tau,
            shape=(*batch_shape, self.n_states - 1), **beta_kwargs,
        ).sort(axis=-1)

        idx = np.arange(n_timepoints)
        tau = pm.Deterministic(
            "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)
        return tau


def stick_breaking(beta, batch_shape=()):
    """Stick-breaking construction turning independent Beta draws into a
    set of weights that sum to 1 (used by Dirichlet-process priors).

    Args:
        beta (pytensor tensor): shape batch_shape + (max_states,)
        batch_shape (tuple): leading batch dimensions.

    Returns:
        pytensor tensor, same shape as beta.
    """
    ones = np.ones((*batch_shape, 1))
    return beta * tt.concatenate(
        [ones, tt.extra_ops.cumprod(1 - beta, axis=-1)[..., :-1]], axis=-1)


class DirichletProcessChangepoint(ChangepointPrior):
    """Automatic state-count changepoint prior via a Dirichlet process
    (stick-breaking) construction.
    """

    def __init__(self, max_states, alpha_a=10, alpha_b=1.5):
        """
        Args:
            max_states (int): maximum number of states to allow (the
                Dirichlet process can effectively use fewer).
            alpha_a (float): shape parameter for the Gamma prior on the
                Gamma hyperprior generating the concentration parameter.
            alpha_b (float): shape parameter for the second Gamma
                hyperprior.
        """
        self.max_states = max_states
        self.alpha_a = alpha_a
        self.alpha_b = alpha_b

    @property
    def n_output_states(self):
        return self.max_states

    def build_tau(self, batch_shape, n_timepoints):
        a_gamma = pm.Gamma("a_gamma", self.alpha_a, 1)
        b_gamma = pm.Gamma("b_gamma", self.alpha_b, 1)
        alpha = pm.Gamma("alpha", a_gamma, b_gamma)
        beta = pm.Beta(
            "beta", 1, alpha, shape=(*batch_shape, self.max_states))
        w_raw = pm.Deterministic(
            "w_raw", stick_breaking(beta, batch_shape))
        w_latent = pm.Deterministic(
            "w_latent", w_raw / w_raw.sum(axis=-1, keepdims=True))
        tau = pm.Deterministic(
            "tau", tt.cumsum(w_latent * n_timepoints, axis=-1)[..., :-1])
        return tau
