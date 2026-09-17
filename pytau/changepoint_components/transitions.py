"""
Shared transition-weight construction for composable changepoint models.

blend_weights builds the "weight_stack" tensor used to blend per-state
emission parameters into a smooth, time-varying value. It is generalized
over an arbitrary leading batch shape (e.g. () for unbatched 2D data,
(trials,) for per-trial models) so the same sigmoid-blend logic that used
to be copy-pasted into every model class lives in exactly one place.
"""

import numpy as np
import pytensor.tensor as tt


def blend_weights(tau, batch_shape, n_timepoints):
    """Build a categorical-blend weight stack from changepoint positions.

    Reproduces the sigmoid weight_stack/inverse_stack construction that was
    previously duplicated across ~12 model classes, generalized over an
    arbitrary leading batch shape.

    Args:
        tau (pytensor tensor): changepoint positions, shape
            batch_shape + (n_states - 1,)
        batch_shape (tuple): leading batch dimensions of tau (e.g. () for
            unbatched data, (trials,) for per-trial models).
        n_timepoints (int): number of time bins.

    Returns:
        weight_stack (pytensor tensor): shape
            batch_shape + (n_states, n_timepoints). Sums to 1 along the
            n_states axis at every timepoint (a soft one-hot blend across
            states).
    """
    idx = np.arange(n_timepoints)
    # idx: (n_timepoints,); tau[..., None]: batch_shape + (n_states-1, 1)
    # broadcasts to batch_shape + (n_states-1, n_timepoints)
    raw = tt.math.sigmoid(idx - tau[..., None])

    ones = np.ones((*batch_shape, 1, n_timepoints))
    weight_stack = tt.concatenate([ones, raw], axis=-2)
    inverse_stack = tt.concatenate(
        [1 - weight_stack[..., 1:, :], ones], axis=-2)
    return weight_stack * inverse_stack
