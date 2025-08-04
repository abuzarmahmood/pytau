"""
PyMC3 Blackbox Variational Inference implementation
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
import numpy as np
import os
import time
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
        """Generate PyMC3 model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_model()")

    def test(self):
        """Test model functionality - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement test()")


############################################################
# Functions
############################################################


def theano_lock_present():
    """
    Check if theano compilation lock is present
    """
    return os.path.exists(os.path.join(theano.config.compiledir, "lock_dir"))


def compile_wait():
    """
    Function to allow waiting while a model is already fitting
    Wait twice because lock blips out between steps
    10 secs of waiting shouldn't be a problem for long fits (~mins)
    And wait a random time in the beginning to stagger fits
    """
    time.sleep(np.random.random() * 10)
    while theano_lock_present():
        print("Lock present...waiting")
        time.sleep(10)
    while theano_lock_present():
        print("Lock present...waiting")
        time.sleep(10)


def gen_test_array(array_size, n_states, type="poisson"):
    """
    Generate test array for model fitting
    Last 2 dimensions consist of a single trial
    Time will always be last dimension

    Args:
        array_size (tuple): Size of array to generate
        n_states (int): Number of states to generate
        type (str): Type of data to generate
            - normal
            - poisson
    """
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
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
            mu = pm.Normal("mu", mu=mean_vals, sigma=1, shape=(y_dim, n_states))
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
        self.data_array = data_array
        self.max_states = max_states

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
            mu = pm.Normal("mu", mu=mean_vals, sigma=1, shape=(y_dim, n_states))
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
        self.data_array = data_array
        self.max_states = max_states

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
        self.data_array = data_array
        self.n_states = n_states
        self.inds_span = inds_span

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
        self.data_array = data_array
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
        self.data_array = data_array
        self.n_states = n_states
        self.inds_span = inds_span

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
        self.data_array = data_array
        self.switch_components = switch_components
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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
        self.data_array = data_array
        self.switch_components = switch_components
        self.n_states = n_states

    def generate_model(self):
        """
        Returns:
            pymc3 model: Model class containing graph to run inference on
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

            weight_stack = tt.nnet.sigmoid(
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
    test_data_2d = gen_test_array((10, 100), n_states=3, type="normal")
    test_data_3d = gen_test_array((5, 10, 100), n_states=3, type="poisson")
    test_data_4d = gen_test_array((2, 5, 10, 100), n_states=3, type="poisson")

    # Test each model class
    models_to_test = [
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


def find_best_states(data, model_generator, n_fit, n_samples, min_states=2, max_states=10):
    """Convenience function to find best number of states for model

    Args:
        data (array): array on which to run inference
        model_generator (function): function that generates model
        n_fit (int): Number of iterationst to fit the model for
        n_samples (int): Number of samples to draw from fitted model
        min_states (int): Minimum number of states to test
        max_states (int): Maximum number of states to test

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
        model = model_generator(data, n_states)
        model, approx = advi_fit(model, n_fit, n_samples)[:2]
        elbo_values.append(approx.hist[-1])
        model_list.append(model)
    best_model = model_list[np.argmin(elbo_values)]
    return best_model, model_list, elbo_values


def dpp_fit(model, n_chains = 24, n_cores = 1, tune = 500, draws = 500,
            use_numpyro = False):
    """Convenience function to fit DPP model
    """
    if not use_numpyro:
        with model:
            dpp_trace = pm.sample(
                                tune = tune,
                                draws = draws, 
                                  target_accept = 0.95,
                                 chains = n_chains,
                                 cores = n_cores,
                                return_inferencedata=False)
    else:
        with model:
            dpp_trace = pm.sample(
                                nuts_sampler = 'numpyro',
                                tune = tune,
                                draws = draws, 
                                  target_accept = 0.95,
                                 chains = n_chains,
                                 cores = n_cores,
                                return_inferencedata=False)
    return dpp_trace


def advi_fit(model, fit, samples):
    """Convenience function to perform ADVI fit on model

    Args:
        model (pymc3 model): model object to run inference on
        fit (int): Number of iterationst to fit the model for
        samples (int): Number of samples to draw from fitted model

    Returns:
        model: original model on which inference was run,
        approx: fitted model,
        lambda_stack: array containing lambda (emission) values,
        tau_samples,: array containing samples from changepoint distribution
        model.obs.observations: processed array on which fit was run
    """

    with model:
        inference = pm.ADVI("full-rank")
        approx = pm.fit(n=fit, method=inference)
        trace = approx.sample(draws=samples)

    # Extract relevant variables from trace
    tau_samples = trace["tau"]
    if "lambda" in trace.varnames:
        lambda_stack = trace["lambda"].swapaxes(0, 1)
        return model, trace, lambda_stack, tau_samples, model.obs.observations
    if "mu" in trace.varnames:
        mu_stack = trace["mu"].swapaxes(0, 1)
        sigma_stack = trace["sigma"].swapaxes(0, 1)
        return model, trace, mu_stack, sigma_stack, tau_samples, model.obs.observations


def mcmc_fit(model, samples):
    """Convenience function to perform ADVI fit on model

    Args:
        model (pymc3 model): model object to run inference on
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
        trace = pm.sample(draws=samples, **sampler_kwargs)
        trace = trace[::10]

    # Extract relevant variables from trace
    tau_samples = trace["tau"]
    if "lambda" in trace.varnames:
        lambda_stack = trace["lambda"].swapaxes(0, 1)
        return model, trace, lambda_stack, tau_samples, model.obs.observations
    if "mu" in trace.varnames:
        mu_stack = trace["mu"].swapaxes(0, 1)
        sigma_stack = trace["sigma"].swapaxes(0, 1)
        return model, trace, mu_stack, sigma_stack, tau_samples, model.obs.observations


######################################################################
# Run Inference
######################################################################


def run_all_tests():
    """Run tests for all model classes"""
    # Create test data
    test_data_2d = gen_test_array((10, 100), n_states=3, type="normal")
    test_data_3d = gen_test_array((5, 10, 100), n_states=3, type="poisson")
    test_data_4d = gen_test_array((2, 5, 10, 100), n_states=3, type="poisson")

    # Test each model class
    models_to_test = [
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
