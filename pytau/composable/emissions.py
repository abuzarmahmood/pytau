"""
Emission component implementations.
"""

import numpy as np
import pymc as pm
from .base import EmissionComponent


class PoissonEmission(EmissionComponent):
    """Poisson emission rates (lambda parameters)."""
    
    def __init__(self, data_shape, n_states, **kwargs):
        """
        Args:
            data_shape (tuple): Shape of the data
            n_states (int): Number of states
            **kwargs: Additional parameters
        """
        super().__init__(data_shape, n_states, **kwargs)
        
    def generate_emissions(self, data_array):
        """Generate Poisson emission rates.
        
        Args:
            data_array: Input spike count data
            
        Returns:
            lambda_latent: Poisson rate parameters
        """
        # Calculate initial values from data splits
        mean_vals = np.array([
            np.mean(x, axis=-1) 
            for x in np.array_split(data_array, self.n_states, axis=-1)
        ]).T
        
        # Handle different data dimensions
        if len(data_array.shape) == 3:  # trials x neurons x time
            mean_vals = np.mean(mean_vals, axis=1)  # Average over trials
            shape = (data_array.shape[1], self.n_states)  # neurons x states
        elif len(data_array.shape) == 2:  # neurons x time or trials x time
            shape = (data_array.shape[0], self.n_states)  # dim x states
        else:
            shape = (self.n_states,)  # Just states for 1D
            
        mean_vals += 0.01  # Avoid zero starting values
        
        # Exponential prior for Poisson rates
        lambda_latent = pm.Exponential(
            "lambda", 1 / mean_vals, shape=shape)
            
        return lambda_latent


class GaussianEmission(EmissionComponent):
    """Gaussian emission means (mu parameters)."""
    
    def __init__(self, data_shape, n_states, include_variance=False, **kwargs):
        """
        Args:
            data_shape (tuple): Shape of the data
            n_states (int): Number of states
            include_variance (bool): Whether to model variance changes
            **kwargs: Additional parameters
        """
        super().__init__(data_shape, n_states, **kwargs)
        self.include_variance = include_variance
        
    def generate_emissions(self, data_array):
        """Generate Gaussian emission parameters.
        
        Args:
            data_array: Input continuous data
            
        Returns:
            tuple: (mu, sigma) emission parameters
        """
        # Calculate initial values from data splits
        mean_vals = np.array([
            np.mean(x, axis=-1) 
            for x in np.array_split(data_array, self.n_states, axis=-1)
        ]).T
        mean_vals += 0.01  # Avoid zero starting values
        
        # Handle different data dimensions
        if len(data_array.shape) == 3:  # trials x neurons x time
            shape = (data_array.shape[0], data_array.shape[1], self.n_states)
        elif len(data_array.shape) == 2:  # neurons x time
            shape = (data_array.shape[0], self.n_states)
        else:
            shape = (self.n_states,)
            
        # Normal prior for means
        mu = pm.Normal("mu", mu=mean_vals, sigma=1, shape=shape)
        
        if self.include_variance:
            # Model variance changes as well
            if len(data_array.shape) >= 2:
                sigma_shape = (data_array.shape[0], self.n_states)
            else:
                sigma_shape = (self.n_states,)
                
            sigma = pm.HalfCauchy("sigma", 3.0, shape=sigma_shape)
            return mu, sigma
        else:
            # Fixed variance across states
            if len(data_array.shape) >= 2:
                test_std = np.std(data_array, axis=-1)
                sigma = pm.HalfCauchy("sigma", test_std, shape=(data_array.shape[0],))
            else:
                test_std = np.std(data_array)
                sigma = pm.HalfCauchy("sigma", test_std)
            return mu, sigma