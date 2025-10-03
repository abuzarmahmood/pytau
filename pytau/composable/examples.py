"""
Example usage of composable models.

This module provides convenience functions and examples for creating
common model combinations using the composable framework.
"""

from .changepoints import FixedChangepoint, DirichletChangepoint
from .emissions import PoissonEmission, GaussianEmission
from .likelihoods import PoissonLikelihood, GaussianLikelihood
from .models import ComposableModel


def create_poisson_fixed_sigmoid(data_array, n_states):
    """Create Poisson model with fixed changepoints and sigmoid transitions.
    
    Args:
        data_array: Input spike count data
        n_states (int): Number of states
        
    Returns:
        ComposableModel instance
    """
    changepoint_comp = FixedChangepoint(n_states, change_type='sigmoid')
    emission_comp = PoissonEmission(data_array.shape, n_states)
    likelihood_comp = PoissonLikelihood()
    
    return ComposableModel(data_array, changepoint_comp, emission_comp, likelihood_comp)


def create_poisson_fixed_step(data_array, n_states):
    """Create Poisson model with fixed changepoints and step transitions.
    
    Args:
        data_array: Input spike count data
        n_states (int): Number of states
        
    Returns:
        ComposableModel instance
    """
    changepoint_comp = FixedChangepoint(n_states, change_type='step')
    emission_comp = PoissonEmission(data_array.shape, n_states)
    likelihood_comp = PoissonLikelihood()
    
    return ComposableModel(data_array, changepoint_comp, emission_comp, likelihood_comp)


def create_poisson_dirichlet_sigmoid(data_array, max_states=15):
    """Create Poisson model with Dirichlet process changepoints and sigmoid transitions.
    
    Args:
        data_array: Input spike count data
        max_states (int): Maximum number of states
        
    Returns:
        ComposableModel instance
    """
    changepoint_comp = DirichletChangepoint(max_states, change_type='sigmoid')
    emission_comp = PoissonEmission(data_array.shape, max_states)
    likelihood_comp = PoissonLikelihood()
    
    return ComposableModel(data_array, changepoint_comp, emission_comp, likelihood_comp)


def create_gaussian_fixed_sigmoid(data_array, n_states, include_variance=False):
    """Create Gaussian model with fixed changepoints and sigmoid transitions.
    
    Args:
        data_array: Input continuous data
        n_states (int): Number of states
        include_variance (bool): Whether to model variance changes
        
    Returns:
        ComposableModel instance
    """
    changepoint_comp = FixedChangepoint(n_states, change_type='sigmoid')
    emission_comp = GaussianEmission(data_array.shape, n_states, include_variance=include_variance)
    likelihood_comp = GaussianLikelihood()
    
    return ComposableModel(data_array, changepoint_comp, emission_comp, likelihood_comp)


def create_gaussian_dirichlet_step(data_array, max_states=15, include_variance=False):
    """Create Gaussian model with Dirichlet process changepoints and step transitions.
    
    Args:
        data_array: Input continuous data
        max_states (int): Maximum number of states
        include_variance (bool): Whether to model variance changes
        
    Returns:
        ComposableModel instance
    """
    changepoint_comp = DirichletChangepoint(max_states, change_type='step')
    emission_comp = GaussianEmission(data_array.shape, max_states, include_variance=include_variance)
    likelihood_comp = GaussianLikelihood()
    
    return ComposableModel(data_array, changepoint_comp, emission_comp, likelihood_comp)


# Dictionary of available combinations for easy access
MODEL_COMBINATIONS = {
    'poisson_fixed_sigmoid': create_poisson_fixed_sigmoid,
    'poisson_fixed_step': create_poisson_fixed_step,
    'poisson_dirichlet_sigmoid': create_poisson_dirichlet_sigmoid,
    'gaussian_fixed_sigmoid': create_gaussian_fixed_sigmoid,
    'gaussian_dirichlet_step': create_gaussian_dirichlet_step,
}


def create_model(model_type, data_array, **kwargs):
    """Create a model using a predefined combination.
    
    Args:
        model_type (str): Type of model combination
        data_array: Input data
        **kwargs: Additional arguments for the specific model type
        
    Returns:
        ComposableModel instance
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_COMBINATIONS:
        available_types = list(MODEL_COMBINATIONS.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Available types: {available_types}")
        
    return MODEL_COMBINATIONS[model_type](data_array, **kwargs)


def list_available_models():
    """List all available predefined model combinations.
    
    Returns:
        list: Available model type strings
    """
    return list(MODEL_COMBINATIONS.keys())