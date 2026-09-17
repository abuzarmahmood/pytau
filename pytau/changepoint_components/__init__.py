"""
Composable building blocks for changepoint models (issue #21).

This package is being introduced incrementally. Phases 0-2 (transitions,
changepoint priors, and delegation of the first few model classes) are
implemented here; hierarchy/trial-switch/additive-increment/categorical/
random-walk emission families are a documented follow-up (see PR history
and CLAUDE.md).
"""

from .composed import ComposedChangepointModel
from .emissions import EmissionModel, NormalEmission, PoissonEmission
from .priors import ChangepointPrior, DirichletProcessChangepoint, FixedCountChangepoint
from .transitions import blend_weights

__all__ = [
    "ChangepointPrior",
    "FixedCountChangepoint",
    "DirichletProcessChangepoint",
    "EmissionModel",
    "PoissonEmission",
    "NormalEmission",
    "ComposedChangepointModel",
    "blend_weights",
]
