"""
Scikit-learn style API for PyTau changepoint detection models.

This module provides a simplified interface for changepoint detection
that accepts numpy arrays directly, similar to scikit-learn estimators.
"""

import numpy as np
import pymc as pm
from typing import Optional, Union, Dict, Any
import warnings

from . import changepoint_model
from . import changepoint_preprocess


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
        random_state: Optional[int] = None
    ):
        self.model_type = model_type
        self.n_states = n_states
        self.bin_width = bin_width
        self.data_transform = data_transform
        self.inference_method = inference_method
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        # Initialize attributes that will be set during fitting
        self.model_ = None
        self.trace_ = None
        self.is_fitted_ = False
        self._processed_data = None
        
        # Validate parameters
        self._validate_parameters()
    
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
            
        valid_transforms = [None, 'trial_shuffled', 'spike_shuffled', 'simulated']
        if self.data_transform not in valid_transforms:
            raise ValueError(f"data_transform must be one of {valid_transforms}")
            
        valid_inference = ['advi', 'nuts']
        if self.inference_method not in valid_inference:
            raise ValueError(f"inference_method must be one of {valid_inference}")
    
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
        X = np.asarray(X)
        
        if X.ndim != 3:
            raise ValueError("Input data must be 3D array (trials x neurons x time)")
            
        if time_lims is None:
            time_lims = (0, X.shape[-1])
            
        # Apply preprocessing if specified
        if self.data_transform is not None:
            processed_data = changepoint_preprocess.preprocess_single_taste(
                X, time_lims, self.bin_width, self.data_transform
            )
        else:
            processed_data = X
            
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
        if self.model_type == 'single_taste_poisson':
            model_class = changepoint_model.SingleTastePoisson(X, self.n_states)
            return model_class.generate_model()
            
        elif self.model_type == 'single_taste_poisson_varsig':
            model_class = changepoint_model.SingleTastePoissonVarsig(X, self.n_states)
            return model_class.generate_model()
            
        elif self.model_type == 'single_taste_poisson_dirichlet':
            return changepoint_model.single_taste_poisson_dirichlet(X, self.n_states)
            
        elif self.model_type == 'gaussian_changepoint_mean_var_2d':
            # For Gaussian model, we need 2D data (features x time)
            if X.ndim == 3:
                # Average across trials for 2D model
                X_2d = np.mean(X, axis=0)
            else:
                X_2d = X
            model_class = changepoint_model.GaussianChangepointMeanVar2D(X_2d, self.n_states)
            return model_class.generate_model()
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
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
        # Set random seed if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Preprocess data
        self._processed_data = self._preprocess_data(X, time_lims)
        
        # Create model
        self.model_ = self._create_model(self._processed_data)
        
        # Fit model
        with self.model_:
            if self.inference_method == 'advi':
                # Use ADVI for faster approximate inference
                inference = pm.ADVI()
                approx = pm.fit(n=self.n_iterations, method=inference)
                self.trace_ = approx.sample(draws=1000)
            elif self.inference_method == 'nuts':
                # Use NUTS for more accurate but slower inference
                self.trace_ = pm.sample(
                    draws=self.n_iterations,
                    tune=1000,
                    random_seed=self.random_state
                )
        
        self.is_fitted_ = True
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
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
            
        if X is not None:
            # Preprocess new data
            processed_data = self._preprocess_data(X)
        else:
            # Use training data
            processed_data = self._processed_data
            
        # Extract predictions from trace
        predictions = {}
        
        # Get changepoint locations (tau)
        if 'tau' in self.trace_.varnames:
            tau_samples = self.trace_['tau']
            predictions['changepoints'] = np.mean(tau_samples, axis=0)
            predictions['changepoint_std'] = np.std(tau_samples, axis=0)
        
        # Get firing rates (lambda)
        if 'lambda' in self.trace_.varnames:
            lambda_samples = self.trace_['lambda']
            predictions['lambda'] = np.mean(lambda_samples, axis=0)
            predictions['lambda_std'] = np.std(lambda_samples, axis=0)
        
        # Compute state sequence based on changepoints
        if 'changepoints' in predictions:
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
            'random_state': self.random_state
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
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
                
        self._validate_parameters()
        
        # Reset fitted state if parameters changed
        self.is_fitted_ = False
        self.model_ = None
        self.trace_ = None
        
        return self


# Convenience functions for backward compatibility and ease of use
def fit_changepoint_model(
    spike_data: np.ndarray,
    model_type: str = 'single_taste_poisson',
    n_states: int = 2,
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
    **kwargs
        Additional parameters passed to ChangepointDetector
        
    Returns
    -------
    detector : ChangepointDetector
        Fitted changepoint detector
    """
    detector = ChangepointDetector(
        model_type=model_type,
        n_states=n_states,
        **kwargs
    )
    detector.fit(spike_data)
    return detector