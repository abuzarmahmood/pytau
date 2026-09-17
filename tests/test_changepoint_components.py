"""
Tests for the changepoint_components package (composable models, issue #21).

These focus on: (1) fast, isolated numerical checks of the shared
transition/prior building blocks against the original inline formulas, and
(2) structural equivalence checks (same declared variables, same
log-probability at the initial point) between each refactored legacy class
and its composed reimplementation. Full ADVI-trajectory equivalence
(bit-for-bit identical ELBO history) was verified manually during
development -- see the PR description -- but is not re-run here since it's
slow (~15-25s per class) and the fast initial-point logp check already
catches any structural regression.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as tt
import pytest

from pytau.changepoint_components import (
    ComposedChangepointModel,
    DirichletProcessChangepoint,
    FixedCountChangepoint,
    NormalEmission,
    PoissonEmission,
)
from pytau.changepoint_components.priors import stick_breaking
from pytau.changepoint_components.transitions import blend_weights
from pytau.changepoint_model import (
    GaussianChangepointMean2D,
    GaussianChangepointMeanDirichlet,
    GaussianChangepointMeanVar2D,
    PoissonChangepoint1D,
    SingleTastePoisson,
    SingleTastePoissonDirichlet,
    gen_test_array,
)


def test_blend_weights_matches_legacy_batched():
    """blend_weights, batch_shape=(trials,), should match the inline
    sigmoid weight-stack formula used by SingleTastePoisson etc."""
    np.random.seed(0)
    trials, n_states, length = 5, 4, 50
    tau_np = np.sort(np.random.uniform(
        5, length - 5, size=(trials, n_states - 1)), axis=-1)
    idx = np.arange(length)

    tau = tt.as_tensor_variable(tau_np)
    legacy = tt.math.sigmoid(idx[np.newaxis, :] - tau[:, :, np.newaxis])
    legacy = tt.concatenate([np.ones((trials, 1, length)), legacy], axis=1)
    inv = 1 - legacy[:, 1:]
    inv = tt.concatenate([inv, np.ones((trials, 1, length))], axis=1)
    legacy_result = (legacy * inv).eval()

    new_result = blend_weights(
        tt.as_tensor_variable(tau_np), (trials,), length).eval()
    np.testing.assert_allclose(legacy_result, new_result)


def test_blend_weights_matches_legacy_unbatched():
    """blend_weights, batch_shape=(), should match the inline sigmoid
    weight-stack formula used by GaussianChangepointMean2D etc."""
    np.random.seed(1)
    n_states, length = 4, 50
    tau_np = np.sort(np.random.uniform(5, length - 5, size=(n_states - 1,)))
    idx = np.arange(length)

    tau = tt.as_tensor_variable(tau_np)
    legacy = tt.math.sigmoid(idx[np.newaxis, :] - tau[:, np.newaxis])
    legacy = tt.concatenate([np.ones((1, length)), legacy], axis=0)
    inv = 1 - legacy[1:]
    inv = tt.concatenate([inv, np.ones((1, length))], axis=0)
    legacy_result = (legacy * inv).eval()

    new_result = blend_weights(
        tt.as_tensor_variable(tau_np), (), length).eval()
    np.testing.assert_allclose(legacy_result, new_result)


def test_stick_breaking_matches_legacy_unbatched():
    np.random.seed(2)
    beta_np = np.random.uniform(0.1, 0.9, size=(6,))
    beta = tt.as_tensor_variable(beta_np)
    legacy = beta * \
        tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    new = stick_breaking(tt.as_tensor_variable(beta_np), batch_shape=())
    np.testing.assert_allclose(legacy.eval(), new.eval())


def test_stick_breaking_matches_legacy_batched():
    np.random.seed(3)
    trials = 5
    beta_np = np.random.uniform(0.1, 0.9, size=(trials, 6))
    beta = tt.as_tensor_variable(beta_np)
    legacy_portion = tt.concatenate(
        [np.ones((trials, 1)), tt.extra_ops.cumprod(1 - beta, axis=-1)[:, :-1]], axis=-1)
    legacy = beta * legacy_portion
    new = stick_breaking(
        tt.as_tensor_variable(beta_np), batch_shape=(trials,))
    np.testing.assert_allclose(legacy.eval(), new.eval())


def _assert_structural_equivalence(legacy_model, new_model):
    """Same declared variables and same log-probability at the (default)
    initial point -- a fast, deterministic proxy for "these two models are
    the same graph," verified in development against full seeded-ADVI
    trajectory equivalence for every case below."""
    assert set(legacy_model.named_vars) == set(new_model.named_vars)
    ip_legacy = legacy_model.initial_point()
    ip_new = new_model.initial_point()
    assert set(ip_legacy) == set(ip_new)
    logp_legacy = legacy_model.compile_logp()(ip_legacy)
    logp_new = new_model.compile_logp()(ip_new)
    assert logp_legacy == pytest.approx(logp_new)


def test_single_taste_poisson_matches_composed():
    np.random.seed(42)
    data = gen_test_array((5, 10, 100), 3, "poisson")
    legacy_model = SingleTastePoisson(data, 3).generate_model()
    new_model = ComposedChangepointModel(
        data,
        changepoint_prior=FixedCountChangepoint(3, hyperprior="halfcauchy"),
        emission_model=PoissonEmission(3),
        batch_shape=(5,),
    ).generate_model()
    _assert_structural_equivalence(legacy_model, new_model)


def test_gaussian_changepoint_mean_2d_matches_composed():
    np.random.seed(42)
    data = gen_test_array((10, 100), 3, "normal")
    legacy_model = GaussianChangepointMean2D(data, 3).generate_model()
    even_switches = np.linspace(0, 1, 4)[1:-1]
    new_model = ComposedChangepointModel(
        data,
        changepoint_prior=FixedCountChangepoint(
            3, hyperprior="halfcauchy", tau_latent_initval=even_switches),
        emission_model=NormalEmission(3),
        batch_shape=(),
    ).generate_model()
    _assert_structural_equivalence(legacy_model, new_model)


def test_gaussian_changepoint_mean_var_2d_matches_composed():
    np.random.seed(1)
    data = gen_test_array((10, 100), 3, "normal")
    legacy_model = GaussianChangepointMeanVar2D(data, 3).generate_model()
    even_switches = np.linspace(0, 1, 4)[1:-1]
    new_model = ComposedChangepointModel(
        data,
        changepoint_prior=FixedCountChangepoint(
            3, hyperprior="halfcauchy", tau_latent_initval=even_switches),
        emission_model=NormalEmission(3, include_variance=True),
        batch_shape=(),
    ).generate_model()
    _assert_structural_equivalence(legacy_model, new_model)


def test_gaussian_changepoint_mean_dirichlet_matches_composed():
    np.random.seed(2)
    data = gen_test_array((10, 100), 3, "normal")
    legacy_model = GaussianChangepointMeanDirichlet(
        data, max_states=5).generate_model()
    test_std = np.std(data, axis=-1)
    new_model = ComposedChangepointModel(
        data,
        changepoint_prior=DirichletProcessChangepoint(max_states=5),
        emission_model=NormalEmission(
            5, include_variance=False, mean_param_name="lambda",
            mu_prior_sigma=10.0, sigma_prior_scale=test_std,
            combined_mean_name="lambda_"),
        batch_shape=(),
    ).generate_model()
    _assert_structural_equivalence(legacy_model, new_model)


def test_single_taste_poisson_dirichlet_matches_composed():
    np.random.seed(3)
    data = gen_test_array((5, 10, 100), 3, "poisson")
    legacy_model = SingleTastePoissonDirichlet(
        data, max_states=5).generate_model()
    new_model = ComposedChangepointModel(
        data,
        changepoint_prior=DirichletProcessChangepoint(max_states=5),
        emission_model=PoissonEmission(5, combined_rate_name="lambda_"),
        batch_shape=(5,),
    ).generate_model()
    _assert_structural_equivalence(legacy_model, new_model)


def test_poisson_changepoint_1d_matches_composed():
    np.random.seed(4)
    data = gen_test_array(100, 3, "poisson")
    legacy_model = PoissonChangepoint1D(data, 3).generate_model()
    even_switches = np.linspace(0, 1, 4)[1:-1]
    new_model = ComposedChangepointModel(
        data,
        changepoint_prior=FixedCountChangepoint(
            3, hyperprior="halfcauchy", tau_latent_initval=even_switches),
        emission_model=PoissonEmission(3),
        batch_shape=(),
    ).generate_model()
    _assert_structural_equivalence(legacy_model, new_model)


@pytest.mark.slow
def test_refactored_classes_build_and_fit_without_error():
    """Smoke test: the production classes (not the standalone composed
    model) should still build a valid pymc model and run a short ADVI fit
    without error, after delegating generate_model() to
    ComposedChangepointModel.

    Deliberately does not call each class's own .test() method: those hit
    a pre-existing, unrelated bug ("trace.varnames" is a pymc3-era
    MultiTrace API that arviz's InferenceData -- what approx.sample() now
    returns -- doesn't have), reproducible on unmodified master and out of
    scope for this change.
    """
    classes_and_var = [
        (SingleTastePoisson(
            gen_test_array((5, 10, 100), 3, "poisson"), 3), "lambda"),
        (GaussianChangepointMean2D(
            gen_test_array((10, 100), 3, "normal"), 3), "mu"),
        (GaussianChangepointMeanVar2D(
            gen_test_array((10, 100), 3, "normal"), 3), "mu"),
        (GaussianChangepointMeanDirichlet(
            gen_test_array((10, 100), 3, "normal"), max_states=5), "lambda"),
        (SingleTastePoissonDirichlet(
            gen_test_array((5, 10, 100), 3, "poisson"), max_states=5), "lambda"),
        (PoissonChangepoint1D(
            gen_test_array(100, 3, "poisson"), 3), "lambda"),
    ]
    for model_instance, expected_var in classes_and_var:
        model = model_instance.generate_model()
        with model:
            approx = pm.fit(n=10, method=pm.ADVI(), progressbar=False)
            trace = approx.sample(draws=10)
        assert expected_var in trace.posterior
