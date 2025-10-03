"""
Composable changepoint models for PyTau.

This module provides base classes and components that can be combined
to create flexible changepoint models with different:
- Changepoint types (fixed vs Dirichlet process)
- Change types (sigmoid vs step)
- Likelihood types (Gaussian vs Poisson)
- Emission models
"""

from .base import ChangepointComponent, EmissionComponent, LikelihoodComponent
from .changepoints import FixedChangepoint, DirichletChangepoint
from .emissions import PoissonEmission, GaussianEmission
from .likelihoods import PoissonLikelihood, GaussianLikelihood
from .models import ComposableModel
from .examples import (
    create_poisson_fixed_sigmoid,
    create_poisson_fixed_step,
    create_poisson_dirichlet_sigmoid,
    create_gaussian_fixed_sigmoid,
    create_gaussian_dirichlet_step,
    create_model,
    list_available_models
)

__all__ = [
    'ChangepointComponent',
    'EmissionComponent', 
    'LikelihoodComponent',
    'FixedChangepoint',
    'DirichletChangepoint',
    'PoissonEmission',
    'GaussianEmission',
    'PoissonLikelihood',
    'GaussianLikelihood',
    'ComposableModel',
    'create_poisson_fixed_sigmoid',
    'create_poisson_fixed_step',
    'create_poisson_dirichlet_sigmoid',
    'create_gaussian_fixed_sigmoid',
    'create_gaussian_dirichlet_step',
    'create_model',
    'list_available_models'
]