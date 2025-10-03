"""
Changepoint component implementations.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as tt
from .base import ChangepointComponent


def stick_breaking(beta):
    """Stick breaking process for Dirichlet process."""
    portion_remaining = tt.concatenate(
        [[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining


class FixedChangepoint(ChangepointComponent):
    """Fixed number of changepoints with Beta priors."""
    
    def __init__(self, n_states, change_type='sigmoid', **kwargs):
        """
        Args:
            n_states (int): Number of states
            change_type (str): Type of change ('sigmoid' or 'step')
            **kwargs: Additional parameters
        """
        super().__init__(n_states, change_type, **kwargs)
        
    def generate_changepoints(self, time_length):
        """Generate fixed changepoints with Beta priors.
        
        Args:
            time_length (int): Length of time dimension
            
        Returns:
            tuple: (tau, weight_stack)
        """
        if self.n_states == 1:
            # No changepoints needed for single state
            weight_stack = tt.ones((1, time_length))
            return None, weight_stack
            
        idx = np.arange(time_length)
        
        # Beta priors for changepoint locations
        a_tau = pm.HalfCauchy("a_tau", 3.0, shape=self.n_states - 1)
        b_tau = pm.HalfCauchy("b_tau", 3.0, shape=self.n_states - 1)
        
        # Initialize changepoints evenly spaced
        even_switches = np.linspace(0, 1, self.n_states + 1)[1:-1]
        tau_latent = pm.Beta(
            "tau_latent", a_tau, b_tau, 
            initval=even_switches, 
            shape=(self.n_states - 1)
        ).sort(axis=-1)
        
        # Scale to time dimension
        tau = pm.Deterministic(
            "tau", idx.min() + (idx.max() - idx.min()) * tau_latent)
        
        # Generate weight stack
        weight_stack = self._generate_weight_stack(tau, time_length)
        
        return tau, weight_stack


class DirichletChangepoint(ChangepointComponent):
    """Dirichlet process changepoints for automatic state detection."""
    
    def __init__(self, max_states=15, change_type='sigmoid', **kwargs):
        """
        Args:
            max_states (int): Maximum number of states in truncated DP
            change_type (str): Type of change ('sigmoid' or 'step')
            **kwargs: Additional parameters
        """
        super().__init__(max_states, change_type, **kwargs)
        self.max_states = max_states
        
    def generate_changepoints(self, time_length):
        """Generate Dirichlet process changepoints.
        
        Args:
            time_length (int): Length of time dimension
            
        Returns:
            tuple: (tau, weight_stack)
        """
        # Hyperpriors on concentration parameter
        a_gamma = pm.Gamma("a_gamma", 10, 1)
        b_gamma = pm.Gamma("b_gamma", 1.5, 1)
        
        # Concentration parameter for beta
        alpha = pm.Gamma("alpha", a_gamma, b_gamma)
        
        # Draw beta's to calculate stick lengths
        beta = pm.Beta("beta", 1, alpha, shape=self.max_states)
        
        # Calculate stick lengths using stick_breaking process
        w_raw = pm.Deterministic("w_raw", stick_breaking(beta))
        
        # Scale stick lengths to time dimension
        w = pm.Deterministic("w", w_raw * time_length)
        
        # Calculate changepoint locations from stick lengths
        tau = pm.Deterministic("tau", tt.cumsum(w)[:-1])
        
        # Generate weight stack based on change type
        if self.change_type == 'sigmoid':
            weight_stack = self._generate_sigmoid_weights_dp(tau, time_length)
        else:
            weight_stack = self._generate_step_weights_dp(w, time_length)
            
        return tau, weight_stack
        
    def _generate_sigmoid_weights_dp(self, tau, time_length):
        """Generate sigmoid weights for Dirichlet process."""
        idx = np.arange(time_length)
        
        # Sigmoid transitions
        weight_stack = tt.math.sigmoid(
            idx[np.newaxis, :] - tau[:, np.newaxis])
        weight_stack = tt.concatenate(
            [tt.ones((1, time_length)), weight_stack], axis=0)
        inverse_stack = 1 - weight_stack[1:]
        inverse_stack = tt.concatenate(
            [inverse_stack, tt.ones((1, time_length))], axis=0)
        weight_stack = weight_stack * inverse_stack
        
        return weight_stack
        
    def _generate_step_weights_dp(self, w, time_length):
        """Generate step weights for Dirichlet process."""
        # For step changes, weights are determined by stick lengths
        # Each state gets weight proportional to its stick length
        weight_stack = tt.zeros((self.max_states, time_length))
        
        cumulative_time = 0
        for i in range(self.max_states):
            start_time = tt.cast(cumulative_time, 'int32')
            end_time = tt.cast(cumulative_time + w[i], 'int32')
            end_time = tt.minimum(end_time, time_length)
            
            # Set weights for this state's time period
            weight_stack = tt.set_subtensor(
                weight_stack[i, start_time:end_time], 1.0)
            
            cumulative_time += w[i]
            
        return weight_stack