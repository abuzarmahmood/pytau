# CLAUDE.md

Project-specific context for working in this repo. This is not a general Python/PyMC tutorial — it's the set of things that aren't obvious from reading any single file, gathered across several triage/development sessions.

## Repo layout

- The git root is this directory. The actual `pytau` Python package lives at `pytau/` under this root (i.e. `pytau/changepoint_model.py`, `pytau/changepoint_io.py`, etc.) — don't confuse a shell prompt showing a path like `.../pytau/pytau` with a nested duplicate; that's just the package directory inside the repo.
- Core modules: `changepoint_model.py` (PyMC model definitions + `advi_fit`/`find_best_states`/`dpp_fit`), `changepoint_io.py` (`FitHandler`/`DatabaseHandler` — the fit pipeline and model-save/database bookkeeping), `changepoint_analysis.py` (`PklHandler`/`_tau`/`_firing` — loading and analyzing saved fits, plus post-fit diagnostics), `changepoint_preprocess.py`.
- `pytau/changepoint_components/`: composable model building blocks (see below).
- `docs/models.md` is the **canonical per-model documentation** — not the README. README.md was deliberately slimmed down during a JOSS-review pass to just a short feature summary that links out to the docs site; when adding or changing a model class, document it in `docs/models.md`, not README.md.

## Model-save configuration

- `pytau/config/MODEL_SAVE_DIR.params` is a plain-text file read by `changepoint_io.py` to determine the default model-save directory (falls back to `~/.pytau/models` if the file is missing or still the `/path/to/directory` placeholder). This mechanism dates to 2023 and was accidentally bypassed by a later refactor (issue #85) — the code has since been fixed to read it again.
- **Never commit a real, environment-specific path into this file.** It's meant to be either the placeholder or overwritten locally by `how_to/scripts/fit_manually.py`/the walkthrough notebooks at runtime. A previous version of this file had a devcontainer-only absolute path checked in, which would break for anyone else once the code started actually reading it.
- `FitHandler`/`DatabaseHandler` both accept an explicit `model_save_dir` constructor argument to override the default entirely.

## Fit output storage

- `FitHandler.save_fit_output()` writes three files per fit: `<path>.pkl` (full cloudpickle — the live PyMC model + VI approximation + full posterior arrays; large and version-locked to the exact pymc/pytensor build used to fit it), `<path>.npz` (a small numpy-only sidecar with just `tau_array`/`lambda_array`/`processed_spikes`/`elbo_hist`, no pymc/pytensor dependency to read back), and `<path>.info` (JSON metadata).
- `PklHandler(file_path, lightweight=False)`: pass `lightweight=True` to force-load the `.npz` sidecar; it's also auto-selected when no `.pkl` exists but a sidecar does. Otherwise the full `.pkl` is used by default.
- The full `.pkl` is intentionally still the default — some analysis scripts (e.g. `pytau/analyses/bla_sigmoid_strength.py`) call `approx.sample(...)` on the loaded model, which only the full pickle can provide.
- `PklHandler.get_diagnostics()` is opt-in (not run in `__init__`) and runs the post-fit data-quality checks in `changepoint_analysis.py` (`calc_firing_drift_anova`, `calc_state_trial_uniformity`, `calc_collapsed_transitions`, `calc_transition_randomness`). It's opt-in because some scripts construct `PklHandler` in tight loops over hundreds of saved fits and shouldn't pay for 4 statistical tests on every load.

## Composable models (`pytau/changepoint_components/`)

All 16 model classes in `changepoint_model.py` used to independently reimplement the same changepoint-prior + emission-combine logic (heavy copy-paste). This is being incrementally replaced with a component architecture:

- `transitions.py`: `blend_weights()` — the sigmoid weight-stack construction, generalized over an arbitrary leading batch shape (`()` unbatched, `(trials,)` per-trial).
- `priors.py`: `ChangepointPrior` (`FixedCountChangepoint`, `DirichletProcessChangepoint`) — produces `tau` only, not the weight_stack (different emissions need different weight tensors from the same `tau`).
- `emissions.py`: `EmissionModel` (`PoissonEmission`, `NormalEmission`) — per-state parameter priors, combine with weight_stack, likelihood.
- `composed.py`: `ComposedChangepointModel` — orchestrates prior + emission into a full pymc model.

**Existing model classes delegate to these components rather than being replaced** — every class keeps its exact name, constructor, `test()` method, and lowercase module-level wrapper function; only `generate_model()`'s body changes to a short call into `ComposedChangepointModel`. When refactoring another class onto this system, verify equivalence against the pre-refactor code before trusting it: same `model.named_vars`, same `initial_point()` log-probability, and ideally a seeded-ADVI ELBO-trajectory match (`pm.fit(n=..., method=pm.ADVI(random_seed=...))`, compare `.hist`). Watch for legacy per-class hyperparameter/naming inconsistencies that must be preserved as explicit constructor parameters, not silently normalized away — e.g. `GaussianChangepointMeanDirichlet` names its mean parameter `"lambda"` (not `"mu"`), uses a more diffuse prior, and a data-derived sigma scale, all different from the other Gaussian classes.

**Migration status** (as of the PR that introduced this): Phases 0-2 done — `SingleTastePoisson`, `GaussianChangepointMean2D`, `GaussianChangepointMeanVar2D`, `GaussianChangepointMeanDirichlet`, `SingleTastePoissonDirichlet`, `PoissonChangepoint1D`. Not yet migrated: `AllTastePoisson`/`AllTastePoissonVarsigFixed`/`SingleTastePoissonTrialSwitch`/`AllTastePoissonTrialSwitch` (need a `TrialBehavior` component + hierarchical emissions — Phase 3, deliberately paused to avoid colliding with in-flight model-addition PRs), `SingleTastePoissonVarsig`/`SingleTastePoissonVarsigFixed` (additive-increment combine style, not the categorical-blend style `blend_weights` provides), `CategoricalChangepoint2D`, and the `RandomWalkChangepoint*` family (Phase 4). Issue #21 tracks the overall effort.

## Local dev environment quirks

- The base conda environment on the primary dev machine has unrelated third-party pytest plugins installed (`dash`, `pytest-playwright`) that crash on plugin autoload due to missing dependencies (`flask`, `greenlet`) unrelated to this project. Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest ...` to avoid this, rather than assuming a bare `pytest` failure means something is wrong with pytau itself.
- `ChangepointModel.test()` methods (the per-class self-test, not the `tests/` pytest suite) are currently broken on master: `approx.sample()` returns an `arviz.InferenceData` with no `.varnames` attribute (a pymc3-era API). Tracked in issue #230. Don't use a class's own `.test()` method as a smoke test until that's fixed — use the actual `tests/test_changepoint_model.py` suite, or build+fit the model directly and check `"varname" in trace.posterior`.
- `tests/test_changepoint_model.py::test_random_walk_changepoint_elbo_state_selection` is flaky (~2/3 pass rate even on unmodified master, confirmed by repeated runs) due to unseeded ADVI internals. Tracked in issue #231. A red CI run on this specific test, with no RandomWalk-related changes in your diff, is very likely unrelated to your change — rerun before assuming it's a regression.

## Open PR conflict hotspots

Before editing `DatabaseHandler.check_mismatched_paths`/`clear_mismatched_paths` in `changepoint_io.py`, check `gh pr list` first — this exact method has repeatedly had multiple simultaneously-open PRs touching the same few lines (path-normalization fixes, function-splitting refactors, cleanup-on-delete additions all landed around the same time).
