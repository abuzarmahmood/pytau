"""
pymc Blackbox Variational Inference implementation
of Poisson Likelihood Changepoint for spike trains.
"""

import os
import time

import numpy as np

########################################
# Import
########################################
import pymc as pm
import pytensor.tensor as tt
from pymc.variational.callbacks import CheckParametersConvergence
from tqdm import tqdm

############################################################
# Base Model Class
############################################################


class ChangepointModel:
    """Base class for all changepoint models"""

    def __init__(self, **kwargs):
        """Initialize model with keyword arguments"""
        self.kwargs = kwargs

    def generate_model(self):
        """Generate pymc model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_model()")

    def test(self):
        """Test model functionality - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement test()")


############################################################
# Functions
############################################################


def check_data_quality(data_array, array_name="data_array"):
    """
    Check input data for infs and nans that could cause fitting issues.

    Args:
        data_array (numpy.ndarray): Input data array to check
        array_name (str): Name of the array for error messages

    Returns:
        bool: True if data is clean, False if issues found
    """
    data_array = np.asarray(data_array)

    has_nan = np.any(np.isnan(data_array))
    has_inf = np.any(np.isinf(data_array))

    if has_nan or has_inf:
        print("=" * 60)
        print("⚠️  WARNING: DATA QUALITY ISSUES DETECTED ⚠️")
        print("=" * 60)

        if has_nan:
            nan_count = np.sum(np.isnan(data_array))
            print(f"❌ Found {nan_count} NaN values in {array_name}")

        if has_inf:
            inf_count = np.sum(np.isinf(data_array))
            print(f"❌ Found {inf_count} infinite values in {array_name}")

        print("\n🚨 MODEL FITTING MAY FAIL OR PRODUCE UNRELIABLE RESULTS!")
        print("\nRecommended actions:")
        print("  • Remove or interpolate NaN values")
        print("  • Replace infinite values with finite numbers")
        print("  • Check data preprocessing pipeline")
        print("=" * 60)

        return False

    return True


def gen_test_array(array_size, n_states, type="poisson"):
    """
    Generate test array for model fitting
    Last 2 dimensions consist of a single trial
    Time will always be last dimension

    Args:
        array_size (tuple or int): Size of array to generate. If int, generates 1D array.
        n_states (int): Number of states to generate
        type (str): Type of data to generate
            - normal
            - poisson
    """
    # Handle 1D case
    if isinstance(array_size, int):
        assert array_size > n_states, "Array too small for states"
        assert type in [
            "normal", "poisson"], "Invalid type, please use normal or poisson"

        # Generate transition times for 1D case
        transition_times = np.random.random(n_states)
        transition_times = np.cumsum(transition_times)
        transition_times = transition_times / transition_times.max()
        transition_times *= array_size
        transition_times = transition_times.astype(int)

        # Generate state bounds
        state_bounds = np.zeros(n_states + 1, dtype=int)
        state_bounds[1:] = transition_times
        state_bounds[-1] = array_size

        # Generate state rates
        lambda_vals = np.random.exponential(2.0, n_states) + 0.5

        # Generate 1D array
        rate_array = np.zeros(array_size)
        for i in range(n_states):
            start_idx = state_bounds[i]
            end_idx = state_bounds[i + 1]
            rate_array[start_idx:end_idx] = lambda_vals[i]

        if type == "poisson":
            return np.random.poisson(rate_array)
        else:
            return np.random.normal(loc=rate_array, scale=0.1)

    # Handle multi-dimensional case (existing code)
    assert array_size[-1] > n_states, "Array too small for states"
    assert type in [
        "normal", "poisson"], "Invalid type, please use normal or poisson"

    # Generate transition times
    transition_times = np.random.random((*array_size[:-2], n_states))
    transition_times = np.cumsum(transition_times, axis=-1)
    transition_times = transition_times / \
        transition_times.max(axis=-1, keepdims=True)
    transition_times *= array_size[-1]
    transition_times = np.vectorize(int)(transition_times)

    # Generate state bounds
    state_bounds = np.zeros((*array_size[:-2], n_states + 1), dtype=int)
    state_bounds[..., 1:] = transition_times

    # Generate state rates
    lambda_vals = np.random.random((*array_size[:-1], n_states))

    # Generate array
    rate_array = np.zeros(array_size)
    inds = list(np.ndindex(lambda_vals.shape))
    for this_ind in inds:
        this_lambda = lambda_vals[this_ind[:-2]][:, this_ind[-1]]
        this_state_bounds = [
            state_bounds[(*this_ind[:-2], this_ind[-1])],
            state_bounds[(*this_ind[:-2], this_ind[-1] + 1)],
        ]
        rate_array[this_ind[:-2]][:,
                                  slice(*this_state_bounds)] = this_lambda[:, None]

    if type == "poisson":
        return np.random.poisson(rate_array)
    else:
        return np.random.normal(loc=rate_array, scale=0.1)


def gen_random_walk_test_array(
    n_points, n_states, mean_range=(-2.0, 2.0), sigma_range=(0.2, 0.6)
):
    """
    Generate a 1D random walk test array with known changepoints in the
    innovation (step) distribution's mean and variance.

    Args:
        n_points (int): Length of the generated random walk.
        n_states (int): Number of states (segments with distinct
            innovation mean/variance) to generate.
        mean_range (tuple): Range from which per-state innovation means
            are drawn.
        sigma_range (tuple): Range from which per-state innovation
            standard deviations are drawn.

    Returns:
        numpy.ndarray: 1D array of length n_points.
    """
    assert n_points > n_states, "Array too small for states"

    n_steps = n_points - 1

    # Generate transition times for the innovation (step) series
    transition_times = np.random.random(n_states)
    transition_times = np.cumsum(transition_times)
    transition_times = transition_times / transition_times.max()
    transition_times *= n_steps
    transition_times = transition_times.astype(int)

    state_bounds = np.zeros(n_states + 1, dtype=int)
    state_bounds[1:] = transition_times
    state_bounds[-1] = n_steps

    # Alternate signs across states to keep segments well separated,
    # regardless of the randomly drawn magnitude
    mean_vals = np.random.uniform(*mean_range, n_states)
    mean_vals = np.abs(mean_vals) * np.resize([1, -1], n_states)
    sigma_vals = np.random.uniform(*sigma_range, n_states)

    innovations = np.zeros(n_steps)
    for i in range(n_states):
        start_idx = state_bounds[i]
        end_idx = state_bounds[i + 1]
        innovations[start_idx:end_idx] = np.random.normal(
            loc=mean_vals[i], scale=sigma_vals[i], size=end_idx - start_idx
        )

    return np.concatenate([[0.0], np.cumsum(innovations)])


def gen_random_walk_participation_test_array(
    n_points,
    n_states,
    mean_range=(-2.0, 2.0),
    sigma_range=(0.2, 0.6),
    disengaged_states=None,
    sigma_disengaged=1.5,
    missing_frac=0.85,
):
    """
    Generate a 1D random walk test array containing one or more
    "disengaged" (non-participating) states, whose innovations are
    drawn from a zero-mean, high-variance distribution and which are
    mostly missing (NaN) in the returned array.

    Args:
        n_points (int): Length of the generated random walk.
        n_states (int): Number of states (segments) to generate.
        mean_range (tuple): Range from which per-state innovation means
            are drawn for participating (engaged) states.
        sigma_range (tuple): Range from which per-state innovation
            standard deviations are drawn for participating states.
        disengaged_states (list[int] or None): Indices of states (0 to
            n_states - 1) to mark as disengaged/non-participating. If
            None, a single state is chosen at random.
        sigma_disengaged (float): Std of the zero-mean innovation
            distribution used for disengaged states.
        missing_frac (float): Fraction of points within disengaged
            segments to blank out to NaN.

    Returns:
        tuple:
            numpy.ndarray: 1D array of length n_points, with NaNs in
                disengaged segments.
            numpy.ndarray: Boolean array of length n_points - 1 marking
                which innovation steps belong to a disengaged state.
    """
    assert n_points > n_states, "Array too small for states"

    n_steps = n_points - 1

    transition_times = np.random.random(n_states)
    transition_times = np.cumsum(transition_times)
    transition_times = transition_times / transition_times.max()
    transition_times *= n_steps
    transition_times = transition_times.astype(int)

    state_bounds = np.zeros(n_states + 1, dtype=int)
    state_bounds[1:] = transition_times
    state_bounds[-1] = n_steps

    if disengaged_states is None:
        disengaged_states = [np.random.randint(n_states)]

    mean_vals = np.random.uniform(*mean_range, n_states)
    mean_vals = np.abs(mean_vals) * np.resize([1, -1], n_states)
    sigma_vals = np.random.uniform(*sigma_range, n_states)

    innovations = np.zeros(n_steps)
    participation_mask = np.zeros(n_steps, dtype=bool)
    for i in range(n_states):
        start_idx = state_bounds[i]
        end_idx = state_bounds[i + 1]
        if i in disengaged_states:
            innovations[start_idx:end_idx] = np.random.normal(
                loc=0.0, scale=sigma_disengaged, size=end_idx - start_idx
            )
            participation_mask[start_idx:end_idx] = True
        else:
            innovations[start_idx:end_idx] = np.random.normal(
                loc=mean_vals[i], scale=sigma_vals[i], size=end_idx - start_idx
            )

    data_array = np.concatenate([[0.0], np.cumsum(innovations)])

    # Blank out most points within disengaged segments (indices into the
    # length n_points data array correspond to step index + 1)
    nan_candidates = np.where(participation_mask)[0] + 1
    n_to_drop = int(missing_frac * len(nan_candidates))
    if n_to_drop > 0:
        drop_idx = np.random.choice(
            nan_candidates, size=n_to_drop, replace=False)
        data_array[drop_idx] = np.nan

    return data_array, participation_mask


############################################################
# Models
############################################################


class GaussianChangepointMeanVar2D(ChangepointModel):
    """Model for gaussian data on 2D array detecting changes in both
    mean and variance.
    """

    def __init__(self, data_array, n_states, **kwargs):
        """
        Args:
            data_array (2D Numpy array): <dimension> x time
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states

        mean_vals = np.array(
            [np.mean(x, axis=-1)
             for x in np.array_split(data_array, n_states, axis=-1)]
        ).T
        mean_vals += 0.01  # To avoid zero starting prob

        y_dim = data_array.shape[0]
        idx = np.arange(data_array.shape[-1])
        length = idx.max() + 1

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=mean_vals, sigma=1,
                           shape=(y_dim, n_states))
            sigma = pm.HalfCauchy("sigma", 3.0, shape=(y_dim, n_states))

            a_tau = pm.HalfCauchy("a_tau", 3.0, shape=n_states - 1)
            b_tau = pm.HalfCauchy("b_tau", 3.0, shape=n_states - 1)

            even_switches = np.linspace(0, 1, n_states + 1)[1:-1]
            tau_latent = pm.Beta(
                "tau_latent", a_tau, b_tau, initval=even_switches, shape=(n_states - 1)
            ).sort(axis=-1)

            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)

            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((1, length)), weight_stack], axis=0)
            inverse_stack = 1 - weight_stack[1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((1, length))], axis=0)
            weight_stack = np.multiply(weight_stack, inverse_stack)

            mu_latent = mu.dot(weight_stack)
            sigma_latent = sigma.dot(weight_stack)
            observation = pm.Normal(
                "obs", mu=mu_latent, sigma=sigma_latent, observed=data_array)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array(
            (10, 100), n_states=self.n_states, type="normal")

        # Create model with test data
        test_model = GaussianChangepointMeanVar2D(test_data, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "mu" in trace.varnames
        assert "sigma" in trace.varnames
        assert "tau" in trace.varnames

        print("Test for GaussianChangepointMeanVar2D passed")
        return True


# For backward compatibility
def gaussian_changepoint_mean_var_2d(data_array, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = GaussianChangepointMeanVar2D(data_array, n_states, **kwargs)
    return model_class.generate_model()


def stick_breaking(beta):
    portion_remaining = tt.concatenate(
        [[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining


class GaussianChangepointMeanDirichlet(ChangepointModel):
    """Model for gaussian data on 2D array detecting changes only in
    the mean. Number of states determined using dirichlet process prior.
    """

    def __init__(self, data_array, max_states=15, **kwargs):
        """
        Args:
            data_array (2D Numpy array): <dimension> x time
            max_states (int): Max number of states to include in truncated dirichlet process
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.max_states = max_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        max_states = self.max_states

        y_dim = data_array.shape[0]
        idx = np.arange(data_array.shape[-1])
        length = idx.max() + 1

        mean_vals = np.array(
            [np.mean(x, axis=-1)
             for x in np.array_split(data_array, max_states, axis=-1)]
        ).T
        mean_vals += 0.01  # To avoid zero starting prob
        test_std = np.std(data_array, axis=-1)

        with pm.Model() as model:
            # ===================
            # Emissions Variables
            # ===================
            lambda_latent = pm.Normal(
                "lambda", mu=mean_vals, sigma=10, shape=(y_dim, max_states))
            # One variance for each dimension
            sigma = pm.HalfCauchy("sigma", test_std, shape=(y_dim))

            # =====================
            # Changepoint Variables
            # =====================

            # Hyperpriors on alpha
            a_gamma = pm.Gamma("a_gamma", 10, 1)
            b_gamma = pm.Gamma("b_gamma", 1.5, 1)

            # Concentration parameter for beta
            alpha = pm.Gamma("alpha", a_gamma, b_gamma)

            # Draw beta's to calculate stick lengths
            beta = pm.Beta("beta", 1, alpha, shape=max_states)

            # Calculate stick lengths using stick_breaking process
            w_raw = pm.Deterministic("w_raw", stick_breaking(beta))

            # Make sure lengths add to 1, and scale to length of data
            w_latent = pm.Deterministic("w_latent", w_raw / w_raw.sum())
            tau = pm.Deterministic("tau", tt.cumsum(w_latent * length)[:-1])

            # Weight stack to assign lambda's to point in time
            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((1, length)), weight_stack], axis=0)
            inverse_stack = 1 - weight_stack[1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((1, length))], axis=0)
            weight_stack = np.multiply(weight_stack, inverse_stack)

            # Create timeseries for latent variable (mean emission)
            lambda_ = pm.Deterministic(
                "lambda_", tt.tensordot(
                    lambda_latent, weight_stack, axes=(1, 0))
            )
            sigma_latent = sigma.dimshuffle(0, "x")

            # Likelihood for observations
            observation = pm.Normal(
                "obs", mu=lambda_, sigma=sigma_latent, observed=data_array)
        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array((10, 100), n_states=3, type="normal")

        # Create model with test data
        test_model = GaussianChangepointMeanDirichlet(test_data, max_states=5)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "lambda" in trace.varnames
        assert "sigma" in trace.varnames
        assert "tau" in trace.varnames
        assert "w_latent" in trace.varnames

        print("Test for GaussianChangepointMeanDirichlet passed")
        return True


# For backward compatibility
def gaussian_changepoint_mean_dirichlet(data_array, max_states=15, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = GaussianChangepointMeanDirichlet(
        data_array, max_states, **kwargs)
    return model_class.generate_model()


# TODO: Convenience function for taking out non-significant states


class GaussianChangepointMean2D(ChangepointModel):
    """Model for gaussian data on 2D array detecting changes only in
    the mean.
    """

    def __init__(self, data_array, n_states, **kwargs):
        """
        Args:
            data_array (2D Numpy array): <dimension> x time
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states

        mean_vals = np.array(
            [np.mean(x, axis=-1)
             for x in np.array_split(data_array, n_states, axis=-1)]
        ).T
        mean_vals += 0.01  # To avoid zero starting prob

        y_dim = data_array.shape[0]
        idx = np.arange(data_array.shape[-1])
        length = idx.max() + 1

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=mean_vals, sigma=1,
                           shape=(y_dim, n_states))
            # One variance for each dimension
            sigma = pm.HalfCauchy("sigma", 3.0, shape=(y_dim))

            a_tau = pm.HalfCauchy("a_tau", 3.0, shape=n_states - 1)
            b_tau = pm.HalfCauchy("b_tau", 3.0, shape=n_states - 1)

            even_switches = np.linspace(0, 1, n_states + 1)[1:-1]
            tau_latent = pm.Beta(
                "tau_latent", a_tau, b_tau, initval=even_switches, shape=(n_states - 1)
            ).sort(axis=-1)

            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)

            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((1, length)), weight_stack], axis=0)
            inverse_stack = 1 - weight_stack[1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((1, length))], axis=0)
            weight_stack = np.multiply(weight_stack, inverse_stack)

            mu_latent = mu.dot(weight_stack)
            sigma_latent = sigma.dimshuffle(0, "x")
            observation = pm.Normal(
                "obs", mu=mu_latent, sigma=sigma_latent, observed=data_array)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array(
            (10, 100), n_states=self.n_states, type="normal")

        # Create model with test data
        test_model = GaussianChangepointMean2D(test_data, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "mu" in trace.varnames
        assert "sigma" in trace.varnames
        assert "tau" in trace.varnames

        print("Test for GaussianChangepointMean2D passed")
        return True


# For backward compatibility
def gaussian_changepoint_mean_2d(data_array, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = GaussianChangepointMean2D(data_array, n_states, **kwargs)
    return model_class.generate_model()


def stick_breaking_trial(this_beta, trial_count):
    portion_remaining = tt.concatenate(
        [
            np.ones((trial_count, 1)),
            tt.extra_ops.cumprod(1 - this_beta, axis=-1)[:, :-1],
        ],
        axis=-1,
    )
    return this_beta * portion_remaining


class SingleTastePoissonDirichlet(ChangepointModel):
    """
    Model for changepoint on single taste using dirichlet process prior
    """

    def __init__(self, data_array, max_states=10, **kwargs):
        """
        Args:
            data_array (3D Numpy array): trials x neurons x time
            max_states (int): Maximum number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.max_states = max_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        max_states = self.max_states

        mean_vals = np.array(
            [np.mean(x, axis=-1)
             for x in np.array_split(data_array, max_states, axis=-1)]
        ).T
        mean_vals = np.mean(mean_vals, axis=1)
        mean_vals += 0.01  # To avoid zero starting prob

        nrns = data_array.shape[1]
        trials = data_array.shape[0]
        idx = np.arange(data_array.shape[-1])
        length = idx.max() + 1

        with pm.Model() as model:
            # ===================
            # Emissions Variables
            # ===================
            lambda_latent = pm.Exponential(
                "lambda", 1 / mean_vals, shape=(nrns, max_states))

            # =====================
            # Changepoint Variables
            # =====================

            # Hyperpriors on alpha
            a_gamma = pm.Gamma("a_gamma", 10, 1)
            b_gamma = pm.Gamma("b_gamma", 1.5, 1)

            # Concentration parameter for beta
            alpha = pm.Gamma("alpha", a_gamma, b_gamma)

            # Draw beta's to calculate stick lengths
            beta = pm.Beta("beta", 1, alpha, shape=(trials, max_states))

            # Calculate stick lengths using stick_breaking process
            w_raw = pm.Deterministic(
                "w_raw", stick_breaking_trial(beta, trials))

            # Make sure lengths add to 1, and scale to length of data
            w_latent = pm.Deterministic(
                "w_latent", w_raw / w_raw.sum(axis=-1)[:, None])
            tau = pm.Deterministic("tau", tt.cumsum(
                w_latent * length, axis=-1)[:, :-1])

            # =====================
            # Rate over time
            # =====================

            # Weight stack to assign lambda's to point in time
            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, :, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((trials, 1, length)), weight_stack], axis=1)
            inverse_stack = 1 - weight_stack[:, 1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((trials, 1, length))], axis=1)
            # Trials x States x Time
            weight_stack = np.multiply(weight_stack, inverse_stack)

            lambda_ = pm.Deterministic(
                "lambda_",
                tt.tensordot(weight_stack, lambda_latent,
                             [1, 1]).swapaxes(1, 2),
            )

            # =====================
            # Likelihood
            # =====================
            observation = pm.Poisson("obs", lambda_, observed=data_array)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array((5, 10, 100), n_states=3, type="poisson")

        # Create model with test data
        test_model = SingleTastePoissonDirichlet(test_data, max_states=5)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "lambda" in trace.varnames
        assert "tau" in trace.varnames
        assert "w_latent" in trace.varnames

        print("Test for SingleTastePoissonDirichlet passed")
        return True


# For backward compatibility
def single_taste_poisson_dirichlet(data_array, max_states=10, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = SingleTastePoissonDirichlet(data_array, max_states, **kwargs)
    return model_class.generate_model()


class SingleTastePoisson(ChangepointModel):
    """Model for changepoint on single taste

    ** Largely taken from "non_hardcoded_changepoint_test_3d.ipynb"
    ** Note : This model does not have hierarchical structure for emissions
    """

    def __init__(self, data_array, n_states, **kwargs):
        """
        Args:
            data_array (3D Numpy array): trials x neurons x time
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states

        mean_vals = np.array(
            [np.mean(x, axis=-1)
             for x in np.array_split(data_array, n_states, axis=-1)]
        ).T
        mean_vals = np.mean(mean_vals, axis=1)
        mean_vals += 0.01  # To avoid zero starting prob

        nrns = data_array.shape[1]
        trials = data_array.shape[0]
        idx = np.arange(data_array.shape[-1])
        length = idx.max() + 1

        with pm.Model() as model:
            lambda_latent = pm.Exponential(
                "lambda", 1 / mean_vals, shape=(nrns, n_states))

            a_tau = pm.HalfCauchy("a_tau", 3.0, shape=n_states - 1)
            b_tau = pm.HalfCauchy("b_tau", 3.0, shape=n_states - 1)

            even_switches = np.linspace(0, 1, n_states + 1)[1:-1]
            tau_latent = pm.Beta(
                "tau_latent",
                a_tau,
                b_tau,
                # initval=even_switches,
                shape=(trials, n_states - 1),
            ).sort(axis=-1)

            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)

            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, :, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((trials, 1, length)), weight_stack], axis=1)
            inverse_stack = 1 - weight_stack[:, 1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((trials, 1, length))], axis=1)
            weight_stack = np.multiply(weight_stack, inverse_stack)

            lambda_ = tt.tensordot(weight_stack, lambda_latent, [
                                   1, 1]).swapaxes(1, 2)
            observation = pm.Poisson("obs", lambda_, observed=data_array)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array(
            (5, 10, 100), n_states=self.n_states, type="poisson")

        # Create model with test data
        test_model = SingleTastePoisson(test_data, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "lambda" in trace.varnames
        assert "tau" in trace.varnames

        print("Test for SingleTastePoisson passed")
        return True


# For backward compatibility
def single_taste_poisson(data_array, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = SingleTastePoisson(data_array, n_states, **kwargs)
    return model_class.generate_model()


def var_sig_exp_tt(x, b):
    """
    x -->
    b -->
    """
    return 1 / (1 + tt.exp(-tt.exp(b) * x))


def var_sig_tt(x, b):
    """
    x -->
    b -->
    """
    return 1 / (1 + tt.exp(-b * x))


class SingleTastePoissonVarsig(ChangepointModel):
    """Model for changepoint on single taste
    **Uses variables sigmoid slope inferred from data

    ** Largely taken from "non_hardcoded_changepoint_test_3d.ipynb"
    ** Note : This model does not have hierarchical structure for emissions
    """

    def __init__(self, data_array, n_states, **kwargs):
        """
        Args:
            data_array (3D Numpy array): trials x neurons x time
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states

        mean_vals = np.array(
            [np.mean(x, axis=-1)
             for x in np.array_split(data_array, n_states, axis=-1)]
        ).T
        mean_vals = np.mean(mean_vals, axis=1)
        mean_vals += 0.01  # To avoid zero starting prob

        lambda_test_vals = np.diff(mean_vals, axis=-1)
        even_switches = np.linspace(0, 1, n_states + 1)[1:-1]

        nrns = data_array.shape[1]
        trials = data_array.shape[0]
        idx = np.arange(data_array.shape[-1])
        length = idx.max() + 1

        with pm.Model() as model:
            # Sigmoid slope
            sig_b = pm.Normal("sig_b", -1, 2, shape=n_states - 1)

            # Initial value
            s0 = pm.Exponential(
                "state0", 1 / (np.mean(mean_vals)), shape=nrns, initval=mean_vals[:, 0]
            )

            # Changes to lambda
            lambda_diff = pm.Normal(
                "lambda_diff",
                mu=0,
                sigma=10,
                shape=(nrns, n_states - 1),
                initval=lambda_test_vals,
            )

            # This is only here to be extracted at the end of sampling
            # NOT USED DIRECTLY IN MODEL
            lambda_fin = pm.Deterministic(
                "lambda", tt.concatenate(
                    [s0[:, np.newaxis], lambda_diff], axis=-1)
            )

            # Changepoint positions
            a = pm.HalfCauchy("a_tau", 10, shape=n_states - 1)
            b = pm.HalfCauchy("b_tau", 10, shape=n_states - 1)

            tau_latent = pm.Beta(
                "tau_latent", a, b,
                # initval=even_switches,
                shape=(trials, n_states - 1)
            ).sort(axis=-1)
            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)

            # Mechanical manipulations to generate firing rates
            idx_temp = np.tile(
                idx[np.newaxis, np.newaxis, :], (trials, n_states - 1, 1))
            tau_temp = tt.tile(tau[:, :, np.newaxis], (1, 1, len(idx)))
            sig_b_temp = tt.tile(
                sig_b[np.newaxis, :, np.newaxis], (trials, 1, len(idx)))

            weight_stack = var_sig_exp_tt(idx_temp - tau_temp, sig_b_temp)
            weight_stack_temp = tt.tile(
                weight_stack[:, np.newaxis, :, :], (1, nrns, 1, 1))

            s0_temp = tt.tile(
                s0[np.newaxis, :, np.newaxis, np.newaxis],
                (trials, 1, n_states - 1, len(idx)),
            )
            lambda_diff_temp = tt.tile(
                lambda_diff[np.newaxis, :, :,
                            np.newaxis], (trials, 1, 1, len(idx))
            )

            # Calculate lambda
            lambda_ = pm.Deterministic(
                "lambda_",
                tt.sum(s0_temp + (weight_stack_temp * lambda_diff_temp), axis=2),
            )
            # Bound lambda to prevent the diffs from making it negative
            # Don't let it go down to zero otherwise we have trouble with probabilities
            lambda_bounded = pm.Deterministic(
                "lambda_bounded", tt.switch(lambda_ >= 0.01, lambda_, 0.01)
            )

            # Add observations
            observation = pm.Poisson(
                "obs", lambda_bounded, observed=data_array)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array(
            (5, 10, 100), n_states=self.n_states, type="poisson")

        # Create model with test data
        test_model = SingleTastePoissonVarsig(test_data, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "lambda" in trace.varnames
        assert "tau" in trace.varnames
        assert "sig_b" in trace.varnames

        print("Test for SingleTastePoissonVarsig passed")
        return True


# For backward compatibility
def single_taste_poisson_varsig(data_array, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = SingleTastePoissonVarsig(data_array, n_states, **kwargs)
    return model_class.generate_model()


def inds_to_b(x_span):
    return 5.8889 / x_span


class SingleTastePoissonVarsigFixed(ChangepointModel):
    """Model for changepoint on single taste
    **Uses sigmoid with given slope

    ** Largely taken from "non_hardcoded_changepoint_test_3d.ipynb"
    ** Note : This model does not have hierarchical structure for emissions
    """

    def __init__(self, data_array, n_states, inds_span=1, **kwargs):
        """
        Args:
            data_array (3D Numpy array): trials x neurons x time
            n_states (int): Number of states to model
            inds_span(float) : Number of indices to cover 5-95% change in sigmoid
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.n_states = n_states
        self.inds_span = inds_span

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states
        inds_span = self.inds_span

        mean_vals = np.array(
            [np.mean(x, axis=-1)
             for x in np.array_split(data_array, n_states, axis=-1)]
        ).T
        mean_vals = np.mean(mean_vals, axis=1)
        mean_vals += 0.01  # To avoid zero starting prob

        lambda_test_vals = np.diff(mean_vals, axis=-1)
        even_switches = np.linspace(0, 1, n_states + 1)[1:-1]

        nrns = data_array.shape[1]
        trials = data_array.shape[0]
        idx = np.arange(data_array.shape[-1])
        length = idx.max() + 1

        # Define sigmoid with given sharpness
        sig_b = inds_to_b(inds_span)

        def sigmoid(x):
            b_temp = tt.tile(
                np.array(sig_b)[None, None, None], x.tag.test_value.shape)
            return 1 / (1 + tt.exp(-b_temp * x))

        with pm.Model() as model:
            # Initial value
            s0 = pm.Exponential(
                "state0", 1 / (np.mean(mean_vals)), shape=nrns, initval=mean_vals[:, 0]
            )

            # Changes to lambda
            lambda_diff = pm.Normal(
                "lambda_diff",
                mu=0,
                sigma=10,
                shape=(nrns, n_states - 1),
                initval=lambda_test_vals,
            )

            # This is only here to be extracted at the end of sampling
            # NOT USED DIRECTLY IN MODEL
            lambda_fin = pm.Deterministic(
                "lambda", tt.concatenate(
                    [s0[:, np.newaxis], lambda_diff], axis=-1)
            )

            # Changepoint positions
            a = pm.HalfCauchy("a_tau", 10, shape=n_states - 1)
            b = pm.HalfCauchy("b_tau", 10, shape=n_states - 1)

            tau_latent = pm.Beta(
                "tau_latent", a, b,
                # initval=even_switches,
                shape=(trials, n_states - 1)
            ).sort(axis=-1)
            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)

            # Mechanical manipulations to generate firing rates
            idx_temp = np.tile(
                idx[np.newaxis, np.newaxis, :], (trials, n_states - 1, 1))
            tau_temp = tt.tile(tau[:, :, np.newaxis], (1, 1, len(idx)))

            weight_stack = sigmoid(idx_temp - tau_temp)
            weight_stack_temp = tt.tile(
                weight_stack[:, np.newaxis, :, :], (1, nrns, 1, 1))

            s0_temp = tt.tile(
                s0[np.newaxis, :, np.newaxis, np.newaxis],
                (trials, 1, n_states - 1, len(idx)),
            )
            lambda_diff_temp = tt.tile(
                lambda_diff[np.newaxis, :, :,
                            np.newaxis], (trials, 1, 1, len(idx))
            )

            # Calculate lambda
            lambda_ = pm.Deterministic(
                "lambda_",
                tt.sum(s0_temp + (weight_stack_temp * lambda_diff_temp), axis=2),
            )
            # Bound lambda to prevent the diffs from making it negative
            # Don't let it go down to zero otherwise we have trouble with probabilities
            lambda_bounded = pm.Deterministic(
                "lambda_bounded", tt.switch(lambda_ >= 0.01, lambda_, 0.01)
            )

            # Add observations
            observation = pm.Poisson(
                "obs", lambda_bounded, observed=data_array)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array(
            (5, 10, 100), n_states=self.n_states, type="poisson")

        # Create model with test data
        test_model = SingleTastePoissonVarsigFixed(
            test_data, self.n_states, self.inds_span)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "lambda" in trace.varnames
        assert "tau" in trace.varnames
        assert "state0" in trace.varnames

        print("Test for SingleTastePoissonVarsigFixed passed")
        return True


# For backward compatibility
def single_taste_poisson_varsig_fixed(data_array, n_states, inds_span=1, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = SingleTastePoissonVarsigFixed(
        data_array, n_states, inds_span, **kwargs)
    return model_class.generate_model()


class AllTastePoisson(ChangepointModel):
    """
    ** Model to fit changepoint to all tastes **
    ** Largely taken from "_v1/poisson_all_tastes_changepoint_model.py"
    """

    def __init__(self, data_array, n_states, **kwargs):
        """
        Args:
            data_array (4D Numpy array): tastes, trials, neurons, time_bins
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states

        # Unroll arrays along taste axis
        data_array_long = np.concatenate(data_array, axis=0)

        # Find mean firing for initial values
        tastes = data_array.shape[0]
        length = data_array.shape[-1]
        nrns = data_array.shape[2]
        trials = data_array.shape[1]

        split_list = np.array_split(data_array, n_states, axis=-1)
        # Cut all to the same size
        min_val = min([x.shape[-1] for x in split_list])
        split_array = np.array([x[..., :min_val] for x in split_list])
        mean_vals = np.mean(split_array, axis=(2, -1)).swapaxes(0, 1)
        mean_vals += 0.01  # To avoid zero starting prob
        mean_nrn_vals = np.mean(mean_vals, axis=(0, 1))

        # Find evenly spaces switchpoints for initial values
        idx = np.arange(data_array.shape[-1])  # Index
        array_idx = np.broadcast_to(idx, data_array_long.shape)
        even_switches = np.linspace(0, idx.max(), n_states + 1)
        even_switches_normal = even_switches / np.max(even_switches)

        taste_label = np.repeat(
            np.arange(data_array.shape[0]), data_array.shape[1])
        trial_num = array_idx.shape[0]

        # Being constructing model
        with pm.Model() as model:
            # Hierarchical firing rates
            # Refer to model diagram
            # Mean firing rate of neuron AT ALL TIMES
            lambda_nrn = pm.Exponential(
                "lambda_nrn", 1 / mean_nrn_vals, shape=(mean_vals.shape[-1])
            )
            # Priors for each state, derived from each neuron
            # Mean firing rate of neuron IN EACH STATE (averaged across tastes)
            lambda_state = pm.Exponential(
                "lambda_state", lambda_nrn, shape=(mean_vals.shape[1:]))
            # Mean firing rate of neuron PER STATE PER TASTE
            lambda_latent = pm.Exponential(
                "lambda",
                lambda_state[np.newaxis, :, :],
                initval=mean_vals,
                shape=(mean_vals.shape),
            )

            # Changepoint time variable
            # INDEPENDENT TAU FOR EVERY TRIAL
            a = pm.HalfNormal("a_tau", 3.0, shape=n_states - 1)
            b = pm.HalfNormal("b_tau", 3.0, shape=n_states - 1)

            # Stack produces n_states x trials --> That gets transposed
            # to trials x n_states and gets sorted along n_states (axis=-1)
            # Sort should work the same way as the Ordered transform -->
            # see rv_sort_test.ipynb
            tau_latent = pm.Beta(
                "tau_latent",
                a,
                b,
                shape=(trial_num, n_states - 1),
                initval=tt.tile(even_switches_normal[1:(
                    n_states)], (array_idx.shape[0], 1)),
            ).sort(axis=-1)

            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)

            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, :, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((tastes * trials, 1, length)), weight_stack], axis=1
            )
            inverse_stack = 1 - weight_stack[:, 1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((tastes * trials, 1, length))], axis=1
            )
            weight_stack = weight_stack * inverse_stack
            weight_stack = tt.tile(
                weight_stack[:, :, None, :], (1, 1, nrns, 1))

            lambda_latent = lambda_latent.dimshuffle(2, 0, 1)
            lambda_latent = tt.repeat(lambda_latent, trials, axis=1)
            lambda_latent = tt.tile(
                lambda_latent[..., None], (1, 1, 1, length))
            lambda_latent = lambda_latent.dimshuffle(1, 2, 0, 3)
            lambda_ = tt.sum(lambda_latent * weight_stack, axis=1)

            observation = pm.Poisson("obs", lambda_, observed=data_array_long)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array(
            (2, 5, 10, 100), n_states=self.n_states, type="poisson")

        # Create model with test data
        test_model = AllTastePoisson(test_data, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "lambda" in trace.varnames
        assert "tau" in trace.varnames
        assert "lambda_nrn" in trace.varnames
        assert "lambda_state" in trace.varnames

        print("Test for AllTastePoisson passed")
        return True


# For backward compatibility
def all_taste_poisson(data_array, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = AllTastePoisson(data_array, n_states, **kwargs)
    return model_class.generate_model()


class AllTastePoissonVarsigFixed(ChangepointModel):
    """
    ** Model to fit changepoint to all tastes with fixed sigmoid **
    ** Largely taken from "_v1/poisson_all_tastes_changepoint_model.py"
    """

    def __init__(self, data_array, n_states, inds_span=1, **kwargs):
        """
        Args:
            data_array (4D Numpy array): tastes, trials, neurons, time_bins
            n_states (int): Number of states to model
            inds_span(float): Number of indices to cover 5-95% change in sigmoid
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.n_states = n_states
        self.inds_span = inds_span

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states
        inds_span = self.inds_span

        # Unroll arrays along taste axis
        data_array_long = np.concatenate(data_array, axis=0)

        # Find mean firing for initial values
        tastes = data_array.shape[0]
        length = data_array.shape[-1]
        nrns = data_array.shape[2]
        trials = data_array.shape[1]

        split_list = np.array_split(data_array, n_states, axis=-1)
        # Cut all to the same size
        min_val = min([x.shape[-1] for x in split_list])
        split_array = np.array([x[..., :min_val] for x in split_list])
        mean_vals = np.mean(split_array, axis=(2, -1)).swapaxes(0, 1)
        mean_vals += 0.01  # To avoid zero starting prob
        mean_nrn_vals = np.mean(mean_vals, axis=(0, 1))

        # Find evenly spaces switchpoints for initial values
        idx = np.arange(data_array.shape[-1])  # Index
        array_idx = np.broadcast_to(idx, data_array_long.shape)
        even_switches = np.linspace(0, idx.max(), n_states + 1)
        even_switches_normal = even_switches / np.max(even_switches)

        taste_label = np.repeat(
            np.arange(data_array.shape[0]), data_array.shape[1])
        trial_num = array_idx.shape[0]

        # Define sigmoid with given sharpness
        sig_b = inds_to_b(inds_span)

        def sigmoid(x):
            b_temp = tt.tile(
                np.array(sig_b)[None, None, None], x.tag.test_value.shape)
            return 1 / (1 + tt.exp(-b_temp * x))

        # Being constructing model
        with pm.Model() as model:
            # Hierarchical firing rates
            # Refer to model diagram
            # Mean firing rate of neuron AT ALL TIMES
            lambda_nrn = pm.Exponential(
                "lambda_nrn", 1 / mean_nrn_vals, shape=(mean_vals.shape[-1])
            )
            # Priors for each state, derived from each neuron
            # Mean firing rate of neuron IN EACH STATE (averaged across tastes)
            lambda_state = pm.Exponential(
                "lambda_state", lambda_nrn, shape=(mean_vals.shape[1:]))
            # Mean firing rate of neuron PER STATE PER TASTE
            lambda_latent = pm.Exponential(
                "lambda",
                lambda_state[np.newaxis, :, :],
                initval=mean_vals,
                shape=(mean_vals.shape),
            )

            # Changepoint time variable
            # INDEPENDENT TAU FOR EVERY TRIAL
            a = pm.HalfNormal("a_tau", 3.0, shape=n_states - 1)
            b = pm.HalfNormal("b_tau", 3.0, shape=n_states - 1)

            # Stack produces n_states x trials --> That gets transposed
            # to trials x n_states and gets sorted along n_states (axis=-1)
            # Sort should work the same way as the Ordered transform -->
            # see rv_sort_test.ipynb
            tau_latent = pm.Beta(
                "tau_latent",
                a,
                b,
                shape=(trial_num, n_states - 1),
                initval=tt.tile(even_switches_normal[1:(
                    n_states)], (array_idx.shape[0], 1)),
            ).sort(axis=-1)

            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)

            weight_stack = sigmoid(idx[np.newaxis, :] - tau[:, :, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((tastes * trials, 1, length)), weight_stack], axis=1
            )
            inverse_stack = 1 - weight_stack[:, 1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((tastes * trials, 1, length))], axis=1
            )
            weight_stack = weight_stack * inverse_stack
            weight_stack = tt.tile(
                weight_stack[:, :, None, :], (1, 1, nrns, 1))

            lambda_latent = lambda_latent.dimshuffle(2, 0, 1)
            lambda_latent = tt.repeat(lambda_latent, trials, axis=1)
            lambda_latent = tt.tile(
                lambda_latent[..., None], (1, 1, 1, length))
            lambda_latent = lambda_latent.dimshuffle(1, 2, 0, 3)
            lambda_ = tt.sum(lambda_latent * weight_stack, axis=1)

            observation = pm.Poisson("obs", lambda_, observed=data_array_long)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array(
            (2, 5, 10, 100), n_states=self.n_states, type="poisson")

        # Create model with test data
        test_model = AllTastePoissonVarsigFixed(
            test_data, self.n_states, self.inds_span)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "lambda" in trace.varnames
        assert "tau" in trace.varnames
        assert "lambda_nrn" in trace.varnames
        assert "lambda_state" in trace.varnames

        print("Test for AllTastePoissonVarsigFixed passed")
        return True


# For backward compatibility
def all_taste_poisson_varsig_fixed(data_array, n_states, inds_span=1, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = AllTastePoissonVarsigFixed(
        data_array, n_states, inds_span, **kwargs)
    return model_class.generate_model()


# def single_taste_poisson_biased_tau_priors(data_array,states):
#     pass

# def single_taste_poisson_hard_padding_tau(data_array,states):
#     pass


class SingleTastePoissonTrialSwitch(ChangepointModel):
    """
    Assuming only emissions change across trials
    Changepoint distribution remains constant
    """

    def __init__(self, data_array, switch_components, n_states, **kwargs):
        """
        Args:
            data_array (3D Numpy array): trials x neurons x time
            switch_components (int): Number of trial switch components
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.switch_components = switch_components
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        switch_components = self.switch_components
        n_states = self.n_states

        trial_num, nrn_num, time_bins = data_array.shape

        with pm.Model() as model:
            # Define Emissions

            # nrns
            nrn_lambda = pm.Exponential("nrn_lambda", 10, shape=(nrn_num))

            # nrns x switch_comps
            trial_lambda = pm.Exponential(
                "trial_lambda",
                nrn_lambda.dimshuffle(0, "x"),
                shape=(nrn_num, switch_components),
            )

            # nrns x switch_comps x n_states
            state_lambda = pm.Exponential(
                "state_lambda",
                trial_lambda.dimshuffle(0, 1, "x"),
                shape=(nrn_num, switch_components, n_states),
            )

            # Define Changepoints
            # Assuming distribution of changepoints remains
            # the same across all trials

            a = pm.HalfCauchy("a_tau", 3.0, shape=n_states - 1)
            b = pm.HalfCauchy("b_tau", 3.0, shape=n_states - 1)

            even_switches = np.linspace(0, 1, n_states + 1)[1:-1]
            tau_latent = pm.Beta(
                "tau_latent", a, b,
                # initval=even_switches,
                shape=(trial_num, n_states - 1)
            ).sort(axis=-1)

            # Trials x Changepoints
            tau = pm.Deterministic("tau", time_bins * tau_latent)

            # Define trial switches
            # Will have same structure as regular changepoints

            even_trial_switches = np.linspace(
                0, 1, switch_components + 1)[1:-1]
            tau_trial_latent = pm.Beta(
                "tau_trial_latent",
                1,
                1,
                initval=even_trial_switches,
                shape=(switch_components - 1),
            ).sort(axis=-1)

            # Trial_changepoints
            tau_trial = pm.Deterministic(
                "tau_trial", trial_num * tau_trial_latent)

            trial_idx = np.arange(trial_num)
            trial_selector = tt.math.sigmoid(
                trial_idx[np.newaxis, :] - tau_trial.dimshuffle(0, "x")
            )

            trial_selector = tt.concatenate(
                [np.ones((1, trial_num)), trial_selector], axis=0)
            inverse_trial_selector = 1 - trial_selector[1:, :]
            inverse_trial_selector = tt.concatenate(
                [inverse_trial_selector, np.ones((1, trial_num))], axis=0
            )

            # First, we can "select" sets of emissions depending on trial_changepoints
            # switch_comps x trials
            trial_selector = np.multiply(
                trial_selector, inverse_trial_selector)

            # state_lambda: nrns x switch_comps x states

            # selected_trial_lambda : nrns x states x trials
            selected_trial_lambda = pm.Deterministic(
                "selected_trial_lambda",
                tt.sum(
                    # "nrns" x switch_comps x "states" x trials
                    trial_selector.dimshuffle("x", 0, "x", 1)
                    * state_lambda.dimshuffle(0, 1, 2, "x"),
                    axis=1,
                ),
            )

            # Then, we can select state_emissions for every trial
            idx = np.arange(time_bins)

            # tau : Trials x Changepoints
            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, :, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((trial_num, 1, time_bins)), weight_stack], axis=1
            )
            inverse_stack = 1 - weight_stack[:, 1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((trial_num, 1, time_bins))], axis=1
            )

            # Trials x states x Time
            weight_stack = np.multiply(weight_stack, inverse_stack)

            # Convert selected_trial_lambda : nrns x trials x states x "time"

            # nrns x trials x time
            lambda_ = tt.sum(
                selected_trial_lambda.dimshuffle(0, 2, 1, "x")
                * weight_stack.dimshuffle("x", 0, 1, 2),
                axis=2,
            )

            # Convert to : trials x nrns x time
            lambda_ = lambda_.dimshuffle(1, 0, 2)

            # Add observations
            observation = pm.Poisson("obs", lambda_, observed=data_array)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array(
            (5, 10, 100), n_states=self.n_states, type="poisson")

        # Create model with test data
        test_model = SingleTastePoissonTrialSwitch(
            test_data, self.switch_components, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "nrn_lambda" in trace.varnames
        assert "tau" in trace.varnames
        assert "tau_trial" in trace.varnames
        assert "state_lambda" in trace.varnames

        print("Test for SingleTastePoissonTrialSwitch passed")
        return True


# For backward compatibility
def single_taste_poisson_trial_switch(data_array, switch_components, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = SingleTastePoissonTrialSwitch(
        data_array, switch_components, n_states, **kwargs)
    return model_class.generate_model()


class AllTastePoissonTrialSwitch(ChangepointModel):
    """
    Assuming only emissions change across trials
    Changepoint distribution remains constant
    """

    def __init__(self, data_array, switch_components, n_states, **kwargs):
        """
        Args:
            data_array (4D Numpy array): tastes, trials, neurons, time_bins
            switch_components (int): Number of trial switch components
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.switch_components = switch_components
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        switch_components = self.switch_components
        n_states = self.n_states

        tastes, trial_num, nrn_num, time_bins = data_array.shape

        with pm.Model() as model:
            # Define Emissions
            # =================================================

            # nrns
            nrn_lambda = pm.Exponential("nrn_lambda", 10, shape=(nrn_num))

            # tastes x nrns
            taste_lambda = pm.Exponential(
                "taste_lambda", nrn_lambda.dimshuffle("x", 0), shape=(tastes, nrn_num)
            )

            # tastes x nrns x switch_comps
            trial_lambda = pm.Exponential(
                "trial_lambda",
                taste_lambda.dimshuffle(0, 1, "x"),
                shape=(tastes, nrn_num, switch_components),
            )

            # tastes x nrns x switch_comps x n_states
            state_lambda = pm.Exponential(
                "state_lambda",
                trial_lambda.dimshuffle(0, 1, 2, "x"),
                shape=(tastes, nrn_num, switch_components, n_states),
            )

            # Define Changepoints
            # =================================================
            # Assuming distribution of changepoints remains
            # the same across all trials

            a = pm.HalfCauchy("a_tau", 3.0, shape=n_states - 1)
            b = pm.HalfCauchy("b_tau", 3.0, shape=n_states - 1)

            even_switches = np.linspace(0, 1, n_states + 1)[1:-1]
            tau_latent = pm.Beta(
                "tau_latent",
                a,
                b,
                # initval=even_switches,
                shape=(tastes, trial_num, n_states - 1),
            ).sort(axis=-1)

            # Tasets x Trials x Changepoints
            tau = pm.Deterministic("tau", time_bins * tau_latent)

            # Define trial switches
            # Will have same structure as regular changepoints

            # a_trial = pm.HalfCauchy('a_trial', 3., shape = switch_components - 1)
            # b_trial = pm.HalfCauchy('b_trial', 3., shape = switch_components - 1)

            even_trial_switches = np.linspace(
                0, 1, switch_components + 1)[1:-1]
            tau_trial_latent = pm.Beta(
                "tau_trial_latent",
                1,
                1,
                initval=even_trial_switches,
                shape=(switch_components - 1),
            ).sort(axis=-1)

            # Trial_changepoints
            # =================================================
            tau_trial = pm.Deterministic(
                "tau_trial", trial_num * tau_trial_latent)

            trial_idx = np.arange(trial_num)
            trial_selector = tt.math.sigmoid(
                trial_idx[np.newaxis, :] - tau_trial.dimshuffle(0, "x")
            )

            trial_selector = tt.concatenate(
                [np.ones((1, trial_num)), trial_selector], axis=0)
            inverse_trial_selector = 1 - trial_selector[1:, :]
            inverse_trial_selector = tt.concatenate(
                [inverse_trial_selector, np.ones((1, trial_num))], axis=0
            )

            # switch_comps x trials
            trial_selector = np.multiply(
                trial_selector, inverse_trial_selector)

            # state_lambda: tastes x nrns x switch_comps x states

            # selected_trial_lambda : tastes x nrns x states x trials
            selected_trial_lambda = pm.Deterministic(
                "selected_trial_lambda",
                tt.sum(
                    # "tastes" x "nrns" x switch_comps x "states" x trials
                    trial_selector.dimshuffle("x", "x", 0, "x", 1)
                    * state_lambda.dimshuffle(0, 1, 2, 3, "x"),
                    axis=2,
                ),
            )

            # First, we can "select" sets of emissions depending on trial_changepoints
            # =================================================
            trial_idx = np.arange(trial_num)
            trial_selector = tt.math.sigmoid(
                trial_idx[np.newaxis, :] - tau_trial.dimshuffle(0, "x")
            )

            trial_selector = tt.concatenate(
                [np.ones((1, trial_num)), trial_selector], axis=0)
            inverse_trial_selector = 1 - trial_selector[1:, :]
            inverse_trial_selector = tt.concatenate(
                [inverse_trial_selector, np.ones((1, trial_num))], axis=0
            )

            # switch_comps x trials
            trial_selector = np.multiply(
                trial_selector, inverse_trial_selector)

            # Then, we can select state_emissions for every trial
            # =================================================

            idx = np.arange(time_bins)

            # tau : Tastes x Trials x Changepoints
            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, :, :, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((tastes, trial_num, 1, time_bins)), weight_stack], axis=2
            )
            inverse_stack = 1 - weight_stack[:, :, 1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((tastes, trial_num, 1, time_bins))], axis=2
            )

            # Tastes x Trials x states x Time
            weight_stack = np.multiply(weight_stack, inverse_stack)

            # Putting everything together
            # =================================================

            # selected_trial_lambda :           tastes x nrns x states x trials
            # Convert selected_trial_lambda --> tastes x trials x nrns x states x "time"

            # weight_stack :           tastes x trials x states x time
            # Convert weight_stack --> tastes x trials x "nrns" x states x time

            # tastes x trials x nrns x time
            lambda_ = tt.sum(
                selected_trial_lambda.dimshuffle(0, 3, 1, 2, "x")
                * weight_stack.dimshuffle(0, 1, "x", 2, 3),
                axis=3,
            )

            # Add observations
            observation = pm.Poisson("obs", lambda_, observed=data_array)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data
        test_data = gen_test_array(
            (2, 5, 10, 100), n_states=self.n_states, type="poisson")

        # Create model with test data
        test_model = AllTastePoissonTrialSwitch(
            test_data, self.switch_components, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "nrn_lambda" in trace.varnames
        assert "tau" in trace.varnames
        assert "tau_trial" in trace.varnames
        assert "state_lambda" in trace.varnames
        assert "taste_lambda" in trace.varnames

        print("Test for AllTastePoissonTrialSwitch passed")
        return True


class CategoricalChangepoint2D(ChangepointModel):
    """Model for categorical data changepoint detection on 2D arrays."""

    def __init__(self, data_array, n_states, **kwargs):
        """
        Args:
            data_array (2D Numpy array): trials x length
                - Each element is a postive integer representing a category
            n_states (int): Number of states to model
            **kwargs: Additional arguments

        """

        super().__init__(**kwargs)
        # Make sure data array is int
        if not np.issubdtype(data_array.dtype, np.integer):
            raise ValueError(
                "Data array must contain integer category values.")
        # Check that data_array is 2D
        if data_array.ndim != 2:
            # If 3D, take the first trial/dimension to make it 2D
            if data_array.ndim == 3:
                data_array = data_array[0]
            else:
                raise ValueError("Data array must be 2D (trials x length).")
        check_data_quality(data_array, "data_array")
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        data_array = self.data_array
        n_states = self.n_states
        trials, length = data_array.shape
        features = len(np.unique(data_array))

        # If features in data_array are not continuous integer values, map them
        feature_set = np.unique(data_array)
        if not np.array_equal(feature_set, np.arange(len(feature_set))):
            # Create a mapping from original categories to continuous integers
            category_map = {cat: i for i, cat in enumerate(feature_set)}
            data_array = np.vectorize(category_map.get)(data_array)

        idx = np.arange(length)
        flat_data_array = data_array.reshape((trials * length,))

        with pm.Model() as model:
            p = pm.Dirichlet("p", a=np.ones(
                (n_states, features)), shape=(n_states, features))

            # Infer changepoint locations
            a_tau = pm.HalfCauchy("a_tau", 3.0, shape=n_states - 1)
            b_tau = pm.HalfCauchy("b_tau", 3.0, shape=n_states - 1)
            # Shape: trials x changepoints
            tau_latent = pm.Beta("tau_latent", a_tau, b_tau, shape=(trials, n_states - 1)).sort(
                axis=-1
            )

            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)

            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, :, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((trials, 1, length)), weight_stack], axis=1)
            inverse_stack = 1 - weight_stack[:, 1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((trials, 1, length))], axis=1)
            weight_stack = np.multiply(weight_stack, inverse_stack)

            # shapes:
            #   - weight_stack: trials x states x length
            #   - p : states x features

            # shape: trials x length x features
            lambda_ = tt.tensordot(weight_stack, p, [1, 0])

            flat_lambda = lambda_.reshape((trials * length, features))

            # Use categorical likelihood
            # data_array = trials x length
            category = pm.Categorical(
                "category", p=flat_lambda, observed=flat_data_array)

        return model

    def test(self):
        test_data = np.random.randint(0, self.n_states, size=(5, 100))
        test_model = CategoricalChangepoint2D(test_data, self.n_states)
        model = test_model.generate_model()
        with model:
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)
        assert "p" in trace.varnames
        assert "tau" in trace.varnames
        print("Test for CategoricalChangepoint2D passed")
        return True


# For backward compatibility
def all_taste_poisson_trial_switch(data_array, switch_components, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = AllTastePoissonTrialSwitch(
        data_array, switch_components, n_states, **kwargs)
    return model_class.generate_model()


######################################################################
# Run Inference
######################################################################


def run_all_tests():
    """Run tests for all model classes"""
    # Create test data
    test_data_1d = gen_test_array(100, n_states=3, type="poisson")
    test_data_2d = gen_test_array((10, 100), n_states=3, type="normal")
    test_data_3d = gen_test_array((5, 10, 100), n_states=3, type="poisson")
    test_data_4d = gen_test_array((2, 5, 10, 100), n_states=3, type="poisson")
    test_data_random_walk = gen_random_walk_test_array(100, n_states=3)
    test_data_participation, _ = gen_random_walk_participation_test_array(
        150, n_states=3)

    # Test each model class
    models_to_test = [
        PoissonChangepoint1D(test_data_1d, 3),
        RandomWalkChangepointMeanVar1D(test_data_random_walk, 3),
        RandomWalkChangepointParticipation1D(test_data_participation, 3),
        RandomWalkChangepointParticipationDirichlet(test_data_participation, max_states=5),
        GaussianChangepointMeanVar2D(test_data_2d, 3),
        GaussianChangepointMeanDirichlet(test_data_2d, 5),
        GaussianChangepointMean2D(test_data_2d, 3),
        SingleTastePoissonDirichlet(test_data_3d, 5),
        SingleTastePoisson(test_data_3d, 3),
        SingleTastePoissonVarsig(test_data_3d, 3),
        SingleTastePoissonVarsigFixed(test_data_3d, 3, 1),
        SingleTastePoissonTrialSwitch(test_data_3d, 2, 3),
        AllTastePoisson(test_data_4d, 3),
        AllTastePoissonVarsigFixed(test_data_4d, 3, 1),
        AllTastePoissonTrialSwitch(test_data_4d, 2, 3),
    ]

    failed_tests = []
    pbar = tqdm(models_to_test, total=len(models_to_test))
    for model in pbar:
        try:
            model.test()
            pbar.set_description(f"Test passed for {model.__class__.__name__}")
        except Exception as e:
            failed_tests.append(model.__class__.__name__)
            print(f"Test failed for {model.__class__.__name__}: {str(e)}")

    print("All tests completed")
    if failed_tests:
        print("Failed tests:", failed_tests)


############################################################
# 1D Poisson Changepoint Model
############################################################


class PoissonChangepoint1D(ChangepointModel):
    """Model for changepoint detection in 1D Poisson time series

    This model detects changepoints in 1D time series data using a Poisson likelihood.
    It assumes the data follows a Poisson distribution with different rates in different
    segments separated by changepoints.
    """

    def __init__(self, data_array, n_states, **kwargs):
        """
        Args:
            data_array (1D Numpy array): Time series data
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.data_array = np.asarray(data_array)
        if self.data_array.ndim != 1:
            raise ValueError("data_array must be 1-dimensional")
        check_data_quality(self.data_array, "data_array")
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states

        # Calculate initial lambda values by splitting data into segments
        mean_vals = np.array([
            np.mean(x) for x in np.array_split(data_array, n_states)
        ])
        mean_vals += 0.01  # To avoid zero starting prob

        idx = np.arange(len(data_array))
        length = len(data_array)

        with pm.Model() as model:
            # Lambda parameters for each state (Poisson rates)
            lambda_latent = pm.Exponential(
                "lambda", 1 / mean_vals, shape=n_states
            )

            # Changepoint locations
            a_tau = pm.HalfCauchy("a_tau", 3.0, shape=n_states - 1)
            b_tau = pm.HalfCauchy("b_tau", 3.0, shape=n_states - 1)

            # Initialize changepoints evenly across the time series
            even_switches = np.linspace(0, 1, n_states + 1)[1:-1]
            tau_latent = pm.Beta(
                "tau_latent",
                a_tau,
                b_tau,
                initval=even_switches,
                shape=(n_states - 1)
            ).sort(axis=-1)

            # Convert to actual time indices
            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent
            )

            # Create weight matrix for smooth transitions between states
            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, np.newaxis]
            )
            weight_stack = tt.concatenate(
                [np.ones((1, length)), weight_stack], axis=0
            )
            inverse_stack = 1 - weight_stack[1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((1, length))], axis=0
            )
            weight_stack = weight_stack * inverse_stack

            # Calculate time-varying lambda
            lambda_t = lambda_latent.dot(weight_stack)

            # Observation model
            observation = pm.Poisson("obs", lambda_t, observed=data_array)

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data - 1D array with 100 time points
        test_data = gen_test_array(100, n_states=self.n_states, type="poisson")

        # Create model with test data
        test_model = PoissonChangepoint1D(test_data, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "lambda" in trace.varnames
        assert "tau" in trace.varnames

        print("Test for PoissonChangepoint1D passed")
        return True


# For backward compatibility
def poisson_changepoint_1d(data_array, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = PoissonChangepoint1D(data_array, n_states, **kwargs)
    return model_class.generate_model()


############################################################
# 1D Random Walk Changepoint Model
############################################################


class RandomWalkChangepointMeanVar1D(ChangepointModel):
    """Model for changepoint detection in 1D random walk time series

    Treats the observed data as a random walk (x_t = x_{t-1} + innovation_t)
    and detects changepoints in both the mean and variance of the
    innovation (step) distribution.
    """

    def __init__(self, data_array, n_states, **kwargs):
        """
        Args:
            data_array (1D Numpy array): Time series data
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.data_array = np.asarray(data_array)
        if self.data_array.ndim != 1:
            raise ValueError("data_array must be 1-dimensional")
        check_data_quality(self.data_array, "data_array")
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states

        # Innovations (steps) of the random walk
        diffs = np.diff(data_array)

        # Calculate initial mean values by splitting innovations into segments
        mean_vals = np.array([
            np.mean(x) for x in np.array_split(diffs, n_states)
        ])

        idx = np.arange(len(diffs))
        length = len(diffs)

        with pm.Model() as model:
            # Mean and variance of the innovation distribution per state
            mu = pm.Normal("mu", mu=mean_vals, sigma=1, shape=n_states)
            sigma = pm.HalfCauchy("sigma", 3.0, shape=n_states)

            # Changepoint locations (over innovation/step indices)
            a_tau = pm.HalfCauchy("a_tau", 3.0, shape=n_states - 1)
            b_tau = pm.HalfCauchy("b_tau", 3.0, shape=n_states - 1)

            # Initialize changepoints evenly across the innovation series
            even_switches = np.linspace(0, 1, n_states + 1)[1:-1]
            tau_latent = pm.Beta(
                "tau_latent",
                a_tau,
                b_tau,
                initval=even_switches,
                shape=(n_states - 1)
            ).sort(axis=-1)

            # Convert to actual step indices
            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent
            )

            # Create weight matrix for smooth transitions between states
            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, np.newaxis]
            )
            weight_stack = tt.concatenate(
                [np.ones((1, length)), weight_stack], axis=0
            )
            inverse_stack = 1 - weight_stack[1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((1, length))], axis=0
            )
            weight_stack = weight_stack * inverse_stack

            # Time-varying innovation mean and variance
            mu_t = mu.dot(weight_stack)
            sigma_t = sigma.dot(weight_stack)

            # Random walk observation model
            init_dist = pm.Normal.dist(mu=data_array[0], sigma=1)
            observation = pm.GaussianRandomWalk(
                "obs",
                mu=mu_t,
                sigma=sigma_t,
                init_dist=init_dist,
                steps=length,
                observed=data_array,
            )

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data - 1D random walk with 100 time points
        test_data = gen_random_walk_test_array(100, n_states=self.n_states)

        # Create model with test data
        test_model = RandomWalkChangepointMeanVar1D(test_data, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "mu" in trace.varnames
        assert "sigma" in trace.varnames
        assert "tau" in trace.varnames

        print("Test for RandomWalkChangepointMeanVar1D passed")
        return True


# For backward compatibility
def random_walk_changepoint_mean_var_1d(data_array, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = RandomWalkChangepointMeanVar1D(
        data_array, n_states, **kwargs)
    return model_class.generate_model()


class RandomWalkChangepointParticipation1D(ChangepointModel):
    """Random walk changepoint model with a soft per-state "participation"
    mixture weight.

    Extends RandomWalkChangepointMeanVar1D to states where the underlying
    process is intermittently absent or disengaged (e.g. an animal not
    licking). Non-participation is modeled as a per-state, changepoint-
    blended mixture weight between the state's normal innovation
    distribution and a zero-mean "disengaged" distribution, rather than
    a hard per-timestep indicator (a discrete latent would not be
    compatible with ADVI, used throughout this module). Missing/sparse
    observations (NaN in data_array) are handled by fitting a full-length
    latent random walk path and only evaluating the likelihood at the
    finite (observed) entries.
    """

    def __init__(self, data_array, n_states, **kwargs):
        """
        Args:
            data_array (1D Numpy array): Time series data, may contain
                NaN at unobserved/non-participating timepoints.
            n_states (int): Number of states to model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.data_array = np.asarray(data_array, dtype=float)
        if self.data_array.ndim != 1:
            raise ValueError("data_array must be 1-dimensional")
        # NaNs are expected here (missing = non-participation), so this
        # class intentionally does not call check_data_quality, which
        # warns/errors on NaNs.
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        n_states = self.n_states

        length = len(data_array) - 1  # number of innovations
        idx = np.arange(length)

        observed_mask = ~np.isnan(data_array)
        observed_idx = np.where(observed_mask)[0]
        assert len(
            observed_idx) > n_states, "Too few observed points for n_states"

        finite_diffs = np.diff(data_array[observed_mask])
        mean_vals = np.array([
            np.mean(x) for x in np.array_split(finite_diffs, n_states)
        ])
        x0_val = float(data_array[observed_idx[0]])

        with pm.Model() as model:
            # Mean and variance of the innovation distribution per state
            mu = pm.Normal("mu", mu=mean_vals, sigma=1, shape=n_states)
            sigma = pm.HalfCauchy("sigma", 3.0, shape=n_states)

            # Per-state probability of "participating" (soft mixture weight)
            participation_prob = pm.Beta(
                "participation_prob", 2.0, 2.0, shape=n_states)
            # Innovation distribution used when disengaged/not participating
            sigma_disengaged = pm.HalfCauchy("sigma_disengaged", 3.0)
            # Small fixed observation noise linking the latent random walk
            # path to the (partially observed) data. Kept fixed rather than
            # inferred: the generative process is a random walk with no
            # separate measurement noise, and letting ADVI infer this value
            # (even under a tight HalfNormal prior) was found to inflate it
            # to absorb the changepoint/participation signal instead of
            # explaining it via mu/sigma/participation_prob.
            obs_sigma = 0.15

            # Changepoint locations (over innovation/step indices) - same
            # pattern as RandomWalkChangepointMeanVar1D
            a_tau = pm.HalfCauchy("a_tau", 3.0, shape=n_states - 1)
            b_tau = pm.HalfCauchy("b_tau", 3.0, shape=n_states - 1)

            even_switches = np.linspace(0, 1, n_states + 1)[1:-1]
            tau_latent = pm.Beta(
                "tau_latent",
                a_tau,
                b_tau,
                initval=even_switches,
                shape=(n_states - 1)
            ).sort(axis=-1)

            tau = pm.Deterministic(
                "tau", idx.min() + (idx.max() - idx.min()) * tau_latent
            )

            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, np.newaxis]
            )
            weight_stack = tt.concatenate(
                [np.ones((1, length)), weight_stack], axis=0
            )
            inverse_stack = 1 - weight_stack[1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((1, length))], axis=0
            )
            weight_stack = weight_stack * inverse_stack

            # Time-varying innovation mean, variance and participation weight
            mu_t = mu.dot(weight_stack)
            sigma_t = sigma.dot(weight_stack)
            p_t = participation_prob.dot(weight_stack)

            # Two-component continuous mixture over innovations: engaged
            # (state-specific mean/variance) vs disengaged (zero-mean,
            # separate variance)
            engaged = pm.Normal.dist(mu=mu_t, sigma=sigma_t)
            disengaged = pm.Normal.dist(mu=0.0, sigma=sigma_disengaged)
            weights = tt.stack([p_t, 1 - p_t], axis=-1)
            innovations = pm.Mixture(
                "innovations", w=weights, comp_dists=[engaged, disengaged], shape=length
            )

            # Full-length latent random walk path, including gaps
            x0 = pm.Normal("x0", mu=x0_val, sigma=1)
            latent_path = pm.Deterministic(
                "latent_path",
                x0 + tt.concatenate([[0.0], tt.cumsum(innovations)]),
            )

            # Only evaluate the likelihood at observed (non-NaN) timepoints
            observation = pm.Normal(
                "obs",
                mu=latent_path[observed_idx],
                sigma=obs_sigma,
                observed=data_array[observed_idx],
            )

        return model

    def test(self):
        """Test the model with synthetic data"""
        # Generate test data - 1D random walk with a disengaged segment
        test_data, _ = gen_random_walk_participation_test_array(
            150, n_states=self.n_states)

        # Create model with test data
        test_model = RandomWalkChangepointParticipation1D(
            test_data, self.n_states)
        model = test_model.generate_model()

        # Run a minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "mu" in trace.varnames
        assert "sigma" in trace.varnames
        assert "tau" in trace.varnames
        assert "participation_prob" in trace.varnames

        print("Test for RandomWalkChangepointParticipation1D passed")
        return True


# For backward compatibility
def random_walk_changepoint_participation_1d(data_array, n_states, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = RandomWalkChangepointParticipation1D(
        data_array, n_states, **kwargs)
    return model_class.generate_model()


class RandomWalkChangepointParticipationDirichlet(ChangepointModel):
    """Truncated Dirichlet-process version of RandomWalkChangepointParticipation1D.

    Combines the stick-breaking changepoint/state-count prior used by
    GaussianChangepointMeanDirichlet with the per-state participation
    mixture and NaN-tolerant missing-data likelihood of
    RandomWalkChangepointParticipation1D, so the number of active states
    (including whether a disengaged/non-participating state is present)
    is inferred rather than fixed in advance.

    Unlike the fixed-n_states model, tau here is a cumulative sum of
    nonnegative stick-breaking weights, so it is automatically
    nondecreasing -- no separate sorted tau_latent/.sort() is needed.

    This model is intended to be fit via many-chain MCMC (see dpp_fit),
    not ADVI: the participation mixture creates a per-timestep,
    locally-multimodal posterior (each innovation is a two-component
    mixture) that ADVI's Gaussian approximations are not well suited to
    represent, and that even NUTS may find difficult on any single
    chain -- many chains are used so that different chains can settle
    into different modes.
    """

    def __init__(self, data_array, max_states=10, **kwargs):
        """
        Args:
            data_array (1D Numpy array): Time series data, may contain
                NaN at unobserved/non-participating timepoints.
            max_states (int): Maximum number of states to include in the
                truncated Dirichlet process.
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.data_array = np.asarray(data_array, dtype=float)
        if self.data_array.ndim != 1:
            raise ValueError("data_array must be 1-dimensional")
        # NaNs are expected here (missing = non-participation), so this
        # class intentionally does not call check_data_quality.
        self.max_states = max_states

    def generate_model(self):
        """
        Returns:
            pymc model: Model class containing graph to run inference on
        """
        data_array = self.data_array
        max_states = self.max_states

        length = len(data_array) - 1  # number of innovations
        idx = np.arange(length)

        observed_mask = ~np.isnan(data_array)
        observed_idx = np.where(observed_mask)[0]
        assert len(observed_idx) > max_states, "Too few observed points for max_states"

        finite_diffs = np.diff(data_array[observed_mask])
        mean_vals = np.array([
            np.mean(x) for x in np.array_split(finite_diffs, max_states)
        ])
        x0_val = float(data_array[observed_idx[0]])

        with pm.Model() as model:
            # Mean and variance of the innovation distribution per state
            mu = pm.Normal("mu", mu=mean_vals, sigma=1, shape=max_states)
            sigma = pm.HalfCauchy("sigma", 3.0, shape=max_states)

            # Per-state probability of "participating" (soft mixture weight)
            participation_prob = pm.Beta(
                "participation_prob", 2.0, 2.0, shape=max_states)
            # Innovation distribution used when disengaged/not participating
            sigma_disengaged = pm.HalfCauchy("sigma_disengaged", 3.0)
            # Small fixed observation noise -- see RandomWalkChangepointParticipation1D
            # for why this is fixed rather than inferred.
            obs_sigma = 0.15

            # Truncated Dirichlet process / stick-breaking prior over
            # changepoint locations (same pattern as
            # GaussianChangepointMeanDirichlet)
            a_gamma = pm.Gamma("a_gamma", 10, 1)
            b_gamma = pm.Gamma("b_gamma", 1.5, 1)
            alpha = pm.Gamma("alpha", a_gamma, b_gamma)
            beta = pm.Beta("beta", 1, alpha, shape=max_states)
            w_raw = stick_breaking(beta)
            w_latent = pm.Deterministic("w_latent", w_raw / w_raw.sum())

            # Cumulative stick length is automatically nondecreasing, so
            # no sorting is needed (unlike the fixed-n_states model's
            # Beta-distributed tau_latent.sort()).
            tau = pm.Deterministic("tau", tt.cumsum(w_latent * length)[:-1])

            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, np.newaxis]
            )
            weight_stack = tt.concatenate(
                [np.ones((1, length)), weight_stack], axis=0
            )
            inverse_stack = 1 - weight_stack[1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((1, length))], axis=0
            )
            weight_stack = weight_stack * inverse_stack

            # Time-varying innovation mean, variance and participation weight
            mu_t = mu.dot(weight_stack)
            sigma_t = sigma.dot(weight_stack)
            p_t = participation_prob.dot(weight_stack)

            # Two-component continuous mixture over innovations: engaged
            # (state-specific mean/variance) vs disengaged (zero-mean,
            # separate variance)
            engaged = pm.Normal.dist(mu=mu_t, sigma=sigma_t)
            disengaged = pm.Normal.dist(mu=0.0, sigma=sigma_disengaged)
            weights = tt.stack([p_t, 1 - p_t], axis=-1)
            innovations = pm.Mixture(
                "innovations", w=weights, comp_dists=[engaged, disengaged], shape=length
            )

            # Full-length latent random walk path, including gaps
            x0 = pm.Normal("x0", mu=x0_val, sigma=1)
            latent_path = pm.Deterministic(
                "latent_path",
                x0 + tt.concatenate([[0.0], tt.cumsum(innovations)]),
            )

            # Only evaluate the likelihood at observed (non-NaN) timepoints
            observation = pm.Normal(
                "obs",
                mu=latent_path[observed_idx],
                sigma=obs_sigma,
                observed=data_array[observed_idx],
            )

        return model

    def test(self):
        """Fast ADVI smoke test (same convention as the other Dirichlet
        models' test() methods) -- real fits should use MCMC, see dpp_fit.
        """
        test_data, _ = gen_random_walk_participation_test_array(
            100, n_states=3)

        test_model = RandomWalkChangepointParticipationDirichlet(
            test_data, max_states=5)
        model = test_model.generate_model()

        with model:
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        assert "mu" in trace.varnames
        assert "sigma" in trace.varnames
        assert "w_latent" in trace.varnames
        assert "participation_prob" in trace.varnames

        print("Test for RandomWalkChangepointParticipationDirichlet passed")
        return True


# For backward compatibility
def random_walk_changepoint_participation_dirichlet(data_array, max_states=10, **kwargs):
    """Wrapper function for backward compatibility"""
    model_class = RandomWalkChangepointParticipationDirichlet(
        data_array, max_states, **kwargs)
    return model_class.generate_model()


def extract_inferred_values(trace):
    """Convenience function to extract inferred values from ADVI fit

    Args:
        trace (dict): trace

    Returns:
        dict: dictionary of inferred values
    """
    # Extract relevant variables from trace
    out_dict = dict(tau_samples=trace["tau"])
    if "lambda" in trace.varnames:
        out_dict["lambda_stack"] = trace["lambda"].swapaxes(0, 1)
    if "mu" in trace.varnames:
        out_dict["mu_stack"] = trace["mu"].swapaxes(0, 1)
        out_dict["sigma_stack"] = trace["sigma"].swapaxes(0, 1)
    return out_dict


def find_best_states(
        data,
        model_generator,
        n_fit, n_samples,
        min_states=2,
        max_states=10,
        convergence_tol=None,
):
    """Convenience function to find best number of states for model

    Args:
        data (array): array on which to run inference
        model_generator (function): function that generates model
        n_fit (int): Number of iterationst to fit the model for
        n_samples (int): Number of samples to draw from fitted model
        min_states (int): Minimum number of states to test
        max_states (int): Maximum number of states to test
        convergence_tol (float): Tolerance for convergence. If None, will not check for convergence.

    Returns:
        best_model: model with best number of states,
        model_list: list of models with different number of states,
        elbo_values: list of elbo values for different number of states
    """
    n_state_array = np.arange(min_states, max_states + 1)
    elbo_values = []
    model_list = []
    for n_states in tqdm(n_state_array):
        print(f"Fitting model with {n_states} states")
        # Have to use int instead of np.int64
        model = model_generator(data, int(n_states))
        model, approx = advi_fit(model, n_fit, n_samples, convergence_tol)[:2]
        elbo_values.append(approx.hist[-1])
        model_list.append(model)
    best_model = model_list[np.argmin(elbo_values)]
    return best_model, model_list, elbo_values


def dpp_fit(model, n_chains=24, n_cores=1, tune=500, draws=500, use_numpyro=False):
    """Convenience function to fit DPP model"""
    if not use_numpyro:
        with model:
            dpp_trace = pm.sample(
                tune=tune,
                draws=draws,
                target_accept=0.95,
                chains=n_chains,
                cores=n_cores,
                return_inferencedata=False,
            )
    else:
        with model:
            dpp_trace = pm.sample(
                nuts_sampler="numpyro",
                tune=tune,
                draws=draws,
                target_accept=0.95,
                chains=n_chains,
                cores=n_cores,
                return_inferencedata=False,
            )
    return dpp_trace


def advi_fit(model, fit, samples, convergence_tol=None):
    """Convenience function to perform ADVI fit on model

    Args:
        model (pymc model): model object to run inference on
        fit (int): Number of iterationst to fit the model for
        samples (int): Number of samples to draw from fitted model

    Returns:
        model: original model on which inference was run,
        approx: fitted model,
        lambda_stack: array containing lambda (emission) values,
        tau_samples,: array containing samples from changepoint distribution
        model.obs.observations: processed array on which fit was run
    """

    if convergence_tol is not None:
        callbacks = [pm.callbacks.CheckParametersConvergence(
            tolerance=convergence_tol)]
        print("Using convergence callback with tolerance:", convergence_tol)
    else:
        callbacks = None
    with model:
        inference = pm.ADVI("full-rank")
        approx = pm.fit(n=fit, method=inference, callbacks=callbacks)

        # Check for inf/nan values in ELBO history
        if hasattr(approx, 'hist') and len(approx.hist) > 0:
            elbo_history = np.array(approx.hist)
            if np.any(np.isnan(elbo_history)) or np.any(np.isinf(elbo_history)):
                print("=" * 80)
                print("⚠️  WARNING: ADVI FIT MAY HAVE FAILED ⚠️")
                print("=" * 80)
                print("ELBO history contains NaN or infinite values!")
                print("This suggests issues with either the model or the data.")
                print("\nRecommended actions:")
                print(
                    "  • Check your data for NaN/inf values using check_data_quality()")
                print("  • Try tracking parameters during fitting to diagnose issues")
                print("  • See: https://www.pymc.io/projects/examples/en/2022.01.0/variational_inference/variational_api_quickstart.html#tracking-parameters")
                print("  • Consider using different priors or model structure")
                print("  • Try MCMC sampling instead of ADVI")
                print("=" * 80)

        idata = approx.sample(draws=samples)

    # Check if tau exists in posterior samples (PyMC5 uses InferenceData)
    if "tau" not in idata.posterior.data_vars:
        available_vars = list(idata.posterior.data_vars.keys())
        raise KeyError(
            f"'tau' not found in posterior samples. Available variables: {available_vars}")

    # Extract relevant variables from InferenceData posterior
    try:
        tau_samples = idata.posterior["tau"].values
        # Handle potential dimension issues
        if tau_samples.ndim > 2:
            tau_samples = tau_samples.reshape(-1, tau_samples.shape[-1])
    except Exception as e:
        print(f"Error extracting tau samples: {e}")
        tau_samples = None

    # Get observed data from model (PyMC5 compatible)
    # Since notebooks don't use fit_data, return None to avoid compatibility issues
    observed_data = None

    if "lambda" in idata.posterior.data_vars:
        try:
            lambda_stack = idata.posterior["lambda"].values
            # Handle potential dimension issues
            if lambda_stack.ndim > 3:
                lambda_stack = lambda_stack.reshape(-1,
                                                    *lambda_stack.shape[-2:])
            lambda_stack = lambda_stack.swapaxes(0, 1)
            return model, approx, lambda_stack, tau_samples, observed_data
        except Exception as e:
            print(f"Error extracting lambda samples: {e}")
            return model, approx, None, tau_samples, observed_data

    if "mu" in idata.posterior.data_vars:
        try:
            mu_stack = idata.posterior["mu"].values
            sigma_stack = idata.posterior["sigma"].values
            # Handle potential dimension issues
            if mu_stack.ndim > 3:
                mu_stack = mu_stack.reshape(-1, *mu_stack.shape[-2:])
            if sigma_stack.ndim > 3:
                sigma_stack = sigma_stack.reshape(-1, *sigma_stack.shape[-2:])
            mu_stack = mu_stack.swapaxes(0, 1)
            sigma_stack = sigma_stack.swapaxes(0, 1)
            return model, approx, mu_stack, sigma_stack, tau_samples, observed_data
        except Exception as e:
            print(f"Error extracting mu/sigma samples: {e}")
            return model, approx, None, None, tau_samples, observed_data

    # Fallback - return what we can
    return model, approx, None, tau_samples, observed_data


def mcmc_fit(model, samples):
    """Convenience function to perform ADVI fit on model

    Args:
        model (pymc model): model object to run inference on
        samples (int): Number of samples to draw using MCMC

    Returns:
        model: original model on which inference was run,
        trace:  samples drawn from MCMC,
        lambda_stack: array containing lambda (emission) values,
        tau_samples,: array containing samples from changepoint distribution
        model.obs.observations: processed array on which fit was run
    """

    with model:
        sampler_kwargs = {"cores": 1, "chains": 4}
        idata = pm.sample(draws=samples, **sampler_kwargs)
        # Thin the samples (every 10th sample)
        idata_thinned = idata.sel(draw=slice(None, None, 10))

    # Extract relevant variables from InferenceData posterior
    try:
        tau_samples = idata_thinned.posterior["tau"].values
        # Handle potential dimension issues
        if tau_samples.ndim > 2:
            tau_samples = tau_samples.reshape(-1, tau_samples.shape[-1])
    except Exception as e:
        print(f"Error extracting tau samples: {e}")
        tau_samples = None

    # Get observed data from model (PyMC5 compatible)
    # Since notebooks don't use fit_data, return None to avoid compatibility issues
    observed_data = None

    if "lambda" in idata_thinned.posterior.data_vars:
        try:
            lambda_stack = idata_thinned.posterior["lambda"].values
            # Handle potential dimension issues
            if lambda_stack.ndim > 3:
                lambda_stack = lambda_stack.reshape(-1,
                                                    *lambda_stack.shape[-2:])
            lambda_stack = lambda_stack.swapaxes(0, 1)
            return model, idata_thinned, lambda_stack, tau_samples, observed_data
        except Exception as e:
            print(f"Error extracting lambda samples: {e}")
            return model, idata_thinned, None, tau_samples, observed_data

    if "mu" in idata_thinned.posterior.data_vars:
        try:
            mu_stack = idata_thinned.posterior["mu"].values
            sigma_stack = idata_thinned.posterior["sigma"].values
            # Handle potential dimension issues
            if mu_stack.ndim > 3:
                mu_stack = mu_stack.reshape(-1, *mu_stack.shape[-2:])
            if sigma_stack.ndim > 3:
                sigma_stack = sigma_stack.reshape(-1, *sigma_stack.shape[-2:])
            mu_stack = mu_stack.swapaxes(0, 1)
            sigma_stack = sigma_stack.swapaxes(0, 1)
            return model, idata_thinned, mu_stack, sigma_stack, tau_samples, observed_data
        except Exception as e:
            print(f"Error extracting mu/sigma samples: {e}")
            return model, idata_thinned, None, None, tau_samples, observed_data

    # Fallback - return what we can
    return model, idata_thinned, None, tau_samples, observed_data
