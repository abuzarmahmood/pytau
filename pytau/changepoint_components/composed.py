"""
Orchestrator that assembles a ChangepointPrior + EmissionModel into a
complete pymc model.
"""

import pymc as pm


class ComposedChangepointModel:
    """Assembles a changepoint prior and an emission model into a complete
    pymc model.

    RV declaration order matches the legacy inline classes exactly
    (emission params first, then changepoint prior, then the weight_stack
    and likelihood), since every existing model class declares them in
    that order and preserving it keeps default-initval jitter behavior as
    close as possible to the pre-refactor models.
    """

    def __init__(self, data_array, changepoint_prior, emission_model, batch_shape):
        """
        Args:
            data_array (Numpy array): model-specific shape (see the
                emission_model's build_params docstring).
            changepoint_prior (ChangepointPrior): produces tau.
            emission_model (EmissionModel): produces per-state params,
                combines them with the weight_stack, and declares the
                likelihood.
            batch_shape (tuple): leading batch dimensions shared by tau and
                the weight_stack (e.g. () or (trials,)).
        """
        self.data_array = data_array
        self.changepoint_prior = changepoint_prior
        self.emission_model = emission_model
        self.batch_shape = batch_shape

    def generate_model(self):
        data_array = self.data_array
        n_timepoints = data_array.shape[-1]

        with pm.Model() as model:
            params = self.emission_model.build_params(
                data_array, self.batch_shape)
            tau = self.changepoint_prior.build_tau(
                self.batch_shape, n_timepoints)
            weight_stack = self.emission_model.build_weight_stack(
                tau, self.batch_shape, n_timepoints)
            time_varying = self.emission_model.combine(
                params, weight_stack, self.batch_shape)
            self.emission_model.likelihood(time_varying, data_array)

        return model
