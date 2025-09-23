"""
Scikit-learn style API for PyTau changepoint detection models.

This module provides a simplified interface for changepoint detection
that accepts numpy arrays directly, similar to scikit-learn estimators.
"""

import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import pymc as pm

from . import changepoint_model, changepoint_preprocess
from .config import verbose_context, verbose_print


class ChangepointDetector:
    """
    Scikit-learn style interface for changepoint detection in neural data.

    This class provides a simplified API that accepts spike train arrays directly
    without requiring HDF5 files, following scikit-learn conventions.

    Parameters
    ----------
    model_type : str, default='single_taste_poisson'
        Type of changepoint model to use. Options:
        - 'single_taste_poisson': Standard Poisson changepoint model
        - 'single_taste_poisson_varsig': Variable sigmoid slope model
        - 'single_taste_poisson_dirichlet': Dirichlet process model
        - 'gaussian_changepoint_mean_var_2d': Gaussian model for 2D data

    n_states : int, default=2
        Number of states/changepoints to detect

    bin_width : int, default=1
        Width for binning spike data (in time units)

    data_transform : str, default=None
        Data transformation to apply. Options:
        - None: Use actual data
        - 'trial_shuffled': Shuffle trials
        - 'spike_shuffled': Shuffle spikes within trials
        - 'simulated': Generate simulated data

    inference_method : str, default='advi'
        Inference method to use. Options:
        - 'advi': Automatic Differentiation Variational Inference
        - 'nuts': No-U-Turn Sampler (MCMC)

    n_iterations : int, default=10000
        Number of iterations for inference

    random_state : int, optional
        Random seed for reproducibility

    verbose : bool, optional
        If True, enable verbose output for this instance only.
        If None, uses global verbose setting.

    Attributes
    ----------
    model_ : pymc.Model
        The fitted PyMC model

    trace_ : pymc.backends.base.MultiTrace
        Posterior samples from inference

    is_fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    >>> import numpy as np
    >>> from pytau import ChangepointDetector
    >>>
    >>> # Generate synthetic spike data (trials x neurons x time)
    >>> spike_data = np.random.poisson(2, size=(20, 5, 100))
    >>>
    >>> # Create and fit model
    >>> detector = ChangepointDetector(n_states=3)
    >>> detector.fit(spike_data)
    >>>
    >>> # Get changepoint predictions
    >>> changepoints = detector.predict(spike_data)
    """

    def __init__(
        self,
        model_type: str = 'single_taste_poisson',
        n_states: int = 2,
        bin_width: int = 1,
        data_transform: Optional[str] = None,
        inference_method: str = 'advi',
        n_iterations: int = 10000,
        random_state: Optional[int] = None,
        verbose: Optional[bool] = None
    ):
        # Set attributes first so _verbose_print can access them
        self.model_type = model_type
        self.n_states = n_states
        self.bin_width = bin_width
        self.data_transform = data_transform
        self.inference_method = inference_method
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.verbose = verbose

        # Now we can use verbose printing
        self._verbose_print(f"🔧 Initializing ChangepointDetector with:")
        self._verbose_print(f"   - Model type: {model_type}")
        self._verbose_print(f"   - Number of states: {n_states}")
        self._verbose_print(f"   - Bin width: {bin_width}")
        self._verbose_print(f"   - Data transform: {data_transform}")
        self._verbose_print(f"   - Inference method: {inference_method}")
        self._verbose_print(f"   - Iterations: {n_iterations}")
        self._verbose_print(f"   - Random state: {random_state}")
        self._verbose_print(f"   - Local verbose: {verbose}")

        # Initialize attributes that will be set during fitting
        self.model_ = None
        self.trace_ = None
        self.is_fitted_ = False
        self._processed_data = None

        # Validate parameters
        self._verbose_print("🔍 Validating parameters...")
        self._validate_parameters()
        self._verbose_print("✅ ChangepointDetector initialized successfully")

    def _validate_parameters(self):
        """Validate initialization parameters."""
        valid_models = [
            'single_taste_poisson',
            'single_taste_poisson_varsig',
            'single_taste_poisson_dirichlet',
            'gaussian_changepoint_mean_var_2d'
        ]
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")

        if self.n_states < 2:
            raise ValueError("n_states must be at least 2")

        if self.bin_width < 1:
            raise ValueError("bin_width must be at least 1")

        valid_transforms = [None, 'trial_shuffled',
                            'spike_shuffled', 'simulated']
        if self.data_transform not in valid_transforms:
            raise ValueError(
                f"data_transform must be one of {valid_transforms}")

        valid_inference = ['advi', 'nuts']
        if self.inference_method not in valid_inference:
            raise ValueError(
                f"inference_method must be one of {valid_inference}")

    def _verbose_print(self, *args, **kwargs):
        """Print function that respects instance or global verbose setting."""
        if self.verbose is not None:
            # Use instance-specific verbose setting
            if self.verbose:
                print(*args, **kwargs)
        else:
            # Use global verbose setting
            verbose_print(*args, **kwargs)

    def _preprocess_data(self, X: np.ndarray, time_lims: Optional[tuple] = None) -> np.ndarray:
        """
        Preprocess spike data for model fitting.

        Parameters
        ----------
        X : array-like of shape (n_trials, n_neurons, n_timepoints)
            Spike train data
        time_lims : tuple, optional
            Time limits for data. If None, uses (0, n_timepoints)

        Returns
        -------
        processed_data : ndarray
            Preprocessed spike data
        """
        self._verbose_print("📊 Preprocessing spike data...")
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError(
                "Input data must be 3D array (trials x neurons x time)")

        self._verbose_print(
            f"   - Input shape: {X.shape} (trials × neurons × time)")

        if time_lims is None:
            time_lims = (0, X.shape[-1])

        self._verbose_print(f"   - Time limits: {time_lims}")
        self._verbose_print(f"   - Bin width: {self.bin_width}")

        # Apply preprocessing if specified
        if self.data_transform is not None:
            self._verbose_print(
                f"   - Applying data transform: {self.data_transform}")
            processed_data = changepoint_preprocess.preprocess_single_taste(
                X, time_lims, self.bin_width, self.data_transform
            )
        else:
            self._verbose_print("   - No data transform applied")
            processed_data = X

        self._verbose_print(f"   - Output shape: {processed_data.shape}")
        self._verbose_print("✅ Data preprocessing completed")
        return processed_data

    def _create_model(self, X: np.ndarray):
        """
        Create the appropriate PyMC model based on model_type.

        Parameters
        ----------
        X : ndarray
            Preprocessed spike data

        Returns
        -------
        model : pymc.Model
            PyMC model ready for inference
        """
        self._verbose_print(f"🏗️  Creating {self.model_type} model...")
        self._verbose_print(f"   - Data shape: {X.shape}")
        self._verbose_print(f"   - Number of states: {self.n_states}")

        if self.model_type == 'single_taste_poisson':
            model_class = changepoint_model.SingleTastePoisson(
                X, self.n_states)
            model = model_class.generate_model()

        elif self.model_type == 'single_taste_poisson_varsig':
            model_class = changepoint_model.SingleTastePoissonVarsig(
                X, self.n_states)
            model = model_class.generate_model()

        elif self.model_type == 'single_taste_poisson_dirichlet':
            model = changepoint_model.single_taste_poisson_dirichlet(
                X, self.n_states)

        elif self.model_type == 'gaussian_changepoint_mean_var_2d':
            # For Gaussian model, we need 2D data (features x time)
            if X.ndim == 3:
                self._verbose_print(
                    "   - Averaging across trials for 2D Gaussian model")
                X_2d = np.mean(X, axis=0)
            else:
                X_2d = X
            self._verbose_print(f"   - 2D data shape: {X_2d.shape}")
            model_class = changepoint_model.GaussianChangepointMeanVar2D(
                X_2d, self.n_states)
            model = model_class.generate_model()

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self._verbose_print("✅ Model created successfully")
        return model

    def fit(self, X: np.ndarray, time_lims: Optional[tuple] = None) -> 'ChangepointDetector':
        """
        Fit the changepoint model to spike data.

        Parameters
        ----------
        X : array-like of shape (n_trials, n_neurons, n_timepoints)
            Spike train data where each element represents spike counts

        time_lims : tuple, optional
            Time limits for analysis. If None, uses full time range

        Returns
        -------
        self : ChangepointDetector
            Returns self for method chaining
        """
        self._verbose_print("🚀 Starting model fitting...")

        # Set random seed if specified
        if self.random_state is not None:
            self._verbose_print(f"🎲 Setting random seed: {self.random_state}")
            np.random.seed(self.random_state)

        # Preprocess data
        self._processed_data = self._preprocess_data(X, time_lims)

        # Create model
        self.model_ = self._create_model(self._processed_data)

        # Fit model
        self._verbose_print(
            f"⚙️  Running {self.inference_method.upper()} inference...")
        self._verbose_print(f"   - Iterations: {self.n_iterations}")

        with self.model_:
            if self.inference_method == 'advi':
                self._verbose_print(
                    "   - Using ADVI (Automatic Differentiation Variational Inference)")
                # Use ADVI for faster approximate inference
                inference = pm.ADVI()
                approx = pm.fit(n=self.n_iterations, method=inference)
                self._verbose_print("   - Sampling from approximation...")
                self.trace_ = approx.sample(draws=1000)
            elif self.inference_method == 'nuts':
                self._verbose_print("   - Using NUTS (No-U-Turn Sampler)")
                # Use NUTS for more accurate but slower inference
                self.trace_ = pm.sample(
                    draws=self.n_iterations,
                    tune=1000,
                    random_seed=self.random_state
                )

        self.is_fitted_ = True
        self._verbose_print("✅ Model fitting completed successfully!")
        return self

    def predict(self, X: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Predict changepoints for given data.

        Parameters
        ----------
        X : array-like of shape (n_trials, n_neurons, n_timepoints), optional
            Spike data to predict on. If None, uses training data.

        Returns
        -------
        predictions : dict
            Dictionary containing:
            - 'changepoints': Mean changepoint locations
            - 'changepoint_std': Standard deviation of changepoint locations
            - 'states': Predicted state sequence
            - 'lambda': Inferred firing rates for each state
        """
        self._verbose_print("🔮 Making predictions...")

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        if X is not None:
            self._verbose_print("   - Using new data for predictions")
            # Preprocess new data
            processed_data = self._preprocess_data(X)
        else:
            self._verbose_print("   - Using training data for predictions")
            # Use training data
            processed_data = self._processed_data

        # Extract predictions from trace
        self._verbose_print(
            "   - Extracting predictions from posterior samples...")
        predictions = {}

        # Get changepoint locations (tau)
        if 'tau' in self.trace_.varnames:
            self._verbose_print("   - Computing changepoint statistics")
            tau_samples = self.trace_['tau']
            predictions['changepoints'] = np.mean(tau_samples, axis=0)
            predictions['changepoint_std'] = np.std(tau_samples, axis=0)
            self._verbose_print(
                f"     - Found {len(predictions['changepoints'])} changepoints")

        # Get firing rates (lambda)
        if 'lambda' in self.trace_.varnames:
            self._verbose_print("   - Computing firing rate statistics")
            lambda_samples = self.trace_['lambda']
            predictions['lambda'] = np.mean(lambda_samples, axis=0)
            predictions['lambda_std'] = np.std(lambda_samples, axis=0)

        # Compute state sequence based on changepoints
        if 'changepoints' in predictions:
            self._verbose_print("   - Computing state sequence")
            n_timepoints = processed_data.shape[-1]
            states = np.zeros(n_timepoints)
            changepoints = predictions['changepoints']

            for i, cp in enumerate(changepoints):
                if i == 0:
                    states[:int(cp)] = 0
                else:
                    prev_cp = changepoints[i-1] if i > 0 else 0
                    states[int(prev_cp):int(cp)] = i

            # Last state
            if len(changepoints) > 0:
                states[int(changepoints[-1]):] = len(changepoints)

            predictions['states'] = states

        self._verbose_print(
            f"✅ Predictions completed! Available keys: {list(predictions.keys())}")
        return predictions

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """
        Return the log-likelihood of the data under the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_trials, n_neurons, n_timepoints)
            Test data
        y : ignored
            Not used, present for API consistency

        Returns
        -------
        score : float
            Log-likelihood of the data
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before scoring")

        processed_data = self._preprocess_data(X)

        with self.model_:
            # Compute log-likelihood using the fitted model
            logp = pm.logp(self.model_.observed_RVs, processed_data)

        return float(np.mean(logp.eval()))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'model_type': self.model_type,
            'n_states': self.n_states,
            'bin_width': self.bin_width,
            'data_transform': self.data_transform,
            'inference_method': self.inference_method,
            'n_iterations': self.n_iterations,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params) -> 'ChangepointDetector':
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : ChangepointDetector
            Returns self for method chaining
        """
        self._verbose_print(f"🔧 Updating parameters: {params}")

        for key, value in params.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                self._verbose_print(f"   - {key}: {old_value} → {value}")
            else:
                raise ValueError(f"Invalid parameter {key}")

        # Re-validate parameters after changes
        self._verbose_print("🔍 Validating updated parameters...")
        self._validate_parameters()

        # Reset fitted state if parameters changed
        if params:  # Only reset if parameters actually changed
            self._verbose_print(
                "🔄 Resetting fitted state due to parameter changes")
            self.is_fitted_ = False
            self.model_ = None
            self.trace_ = None

        self._verbose_print("✅ Parameters updated successfully")
        return self


# Convenience functions for backward compatibility and ease of use
def fit_changepoint_model(
    spike_data: np.ndarray,
    model_type: str = 'single_taste_poisson',
    n_states: int = 2,
    verbose: Optional[bool] = None,
    **kwargs
) -> ChangepointDetector:
    """
    Convenience function to fit a changepoint model with minimal setup.

    Parameters
    ----------
    spike_data : array-like of shape (n_trials, n_neurons, n_timepoints)
        Spike train data
    model_type : str, default='single_taste_poisson'
        Type of model to fit
    n_states : int, default=2
        Number of states to detect
    verbose : bool, optional
        If True, enable verbose output. If None, uses global setting.
    **kwargs
        Additional parameters passed to ChangepointDetector

    Returns
    -------
    detector : ChangepointDetector
        Fitted changepoint detector
    """
    # Handle verbose setting for this function
    if verbose is not None:
        with verbose_context(verbose):
            verbose_print(
                "🎯 Using convenience function to fit changepoint model")
            verbose_print(f"   - Model type: {model_type}")
            verbose_print(f"   - Number of states: {n_states}")
            verbose_print(f"   - Additional parameters: {list(kwargs.keys())}")

            detector = ChangepointDetector(
                model_type=model_type,
                n_states=n_states,
                verbose=verbose,
                **kwargs
            )
            detector.fit(spike_data)

            verbose_print("✅ Convenience function completed successfully")
    else:
        verbose_print("🎯 Using convenience function to fit changepoint model")
        verbose_print(f"   - Model type: {model_type}")
        verbose_print(f"   - Number of states: {n_states}")
        verbose_print(f"   - Additional parameters: {list(kwargs.keys())}")

        detector = ChangepointDetector(
            model_type=model_type,
            n_states=n_states,
            verbose=verbose,
            **kwargs
        )
        detector.fit(spike_data)

        verbose_print("✅ Convenience function completed successfully")

    return detector
