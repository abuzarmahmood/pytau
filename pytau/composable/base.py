"""
Base classes for composable changepoint model components.
"""

from abc import ABC, abstractmethod

import numpy as np
import pymc as pm


class ChangepointComponent(ABC):
    """Base class for changepoint components.

    Handles the generation of changepoint (tau) variables with different
    prior types and change types.
    """

    def __init__(self, n_states, change_type='sigmoid', **kwargs):
        """
        Args:
            n_states (int): Number of states
            change_type (str): Type of change ('sigmoid' or 'step')
            **kwargs: Additional component-specific parameters
        """
        self.n_states = n_states
        self.change_type = change_type
        self.kwargs = kwargs

    @abstractmethod
    def generate_changepoints(self, time_length):
        """Generate changepoint variables.

        Args:
            time_length (int): Length of time dimension

        Returns:
            tuple: (tau, weight_stack) where tau are changepoint times
                   and weight_stack are the weights for each state over time
        """
        pass

    def _generate_weight_stack(self, tau, time_length):
        """Generate weight stack from changepoints.

        Args:
            tau: Changepoint times
            time_length (int): Length of time dimension

        Returns:
            weight_stack: Weights for each state over time
        """
        idx = np.arange(time_length)

        if self.change_type == 'sigmoid':
            # Sigmoid transitions
            import pytensor.tensor as tt
            weight_stack = tt.math.sigmoid(
                idx[np.newaxis, :] - tau[:, np.newaxis])
            weight_stack = tt.concatenate(
                [np.ones((1, time_length)), weight_stack], axis=0)
            inverse_stack = 1 - weight_stack[1:]
            inverse_stack = tt.concatenate(
                [inverse_stack, np.ones((1, time_length))], axis=0)
            weight_stack = weight_stack * inverse_stack

        elif self.change_type == 'step':
            # Step transitions
            import pytensor.tensor as tt
            weight_stack = tt.zeros((self.n_states, time_length))

            # Create step function weights
            for i in range(self.n_states):
                if i == 0:
                    # First state: from 0 to first changepoint
                    if self.n_states > 1:
                        weight_stack = tt.set_subtensor(
                            weight_stack[i, :tau[0]], 1.0)
                    else:
                        weight_stack = tt.set_subtensor(
                            weight_stack[i, :], 1.0)
                elif i == self.n_states - 1:
                    # Last state: from last changepoint to end
                    weight_stack = tt.set_subtensor(
                        weight_stack[i, tau[i-1]:], 1.0)
                else:
                    # Middle states: between changepoints
                    weight_stack = tt.set_subtensor(
                        weight_stack[i, tau[i-1]:tau[i]], 1.0)
        else:
            raise ValueError(f"Unknown change_type: {self.change_type}")

        return weight_stack


class EmissionComponent(ABC):
    """Base class for emission components.

    Handles the generation of emission rate/mean variables for different
    likelihood types.
    """

    def __init__(self, data_shape, n_states, **kwargs):
        """
        Args:
            data_shape (tuple): Shape of the data
            n_states (int): Number of states
            **kwargs: Additional component-specific parameters
        """
        self.data_shape = data_shape
        self.n_states = n_states
        self.kwargs = kwargs

    @abstractmethod
    def generate_emissions(self, data_array):
        """Generate emission variables.

        Args:
            data_array: Input data array

        Returns:
            emission variables (e.g., lambda for Poisson, mu for Gaussian)
        """
        pass


class LikelihoodComponent(ABC):
    """Base class for likelihood components.

    Handles the generation of observation likelihood given emissions
    and changepoint weights.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Additional component-specific parameters
        """
        self.kwargs = kwargs

    @abstractmethod
    def generate_likelihood(self, data_array, emissions, weight_stack):
        """Generate likelihood for observations.

        Args:
            data_array: Observed data
            emissions: Emission variables from EmissionComponent
            weight_stack: Weight stack from ChangepointComponent

        Returns:
            PyMC observation variable
        """
        pass
