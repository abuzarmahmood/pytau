# Understanding PyMC Concepts in PyTau

This guide explains the key PyMC concepts used in PyTau for users who may not be familiar with Bayesian modeling or PyMC. For more detailed information, refer to the [official PyMC documentation](https://www.pymc.io/welcome.html).

## Overview

PyTau uses [PyMC](https://www.pymc.io/), a probabilistic programming library, to perform Bayesian inference on changepoint models. This approach allows us to:

- Quantify uncertainty in changepoint locations
- Incorporate prior knowledge about neural dynamics
- Obtain full posterior distributions rather than point estimates

## Core Concepts

### 1. Model

A **model** in PyMC is a probabilistic representation of your data and assumptions. It consists of:

- **Prior distributions**: What we believe about parameters before seeing data
- **Likelihood**: How the data is generated given the parameters
- **Observed data**: The actual measurements we want to explain

In PyTau, models are defined as classes that inherit from `ChangepointModel` and implement a `generate_model()` method.

#### Example: Single Taste Poisson Model

```python
from pytau.changepoint_model import SingleTastePoisson

# Create a model for spike train data
# spike_array shape: (trials, neurons, time)
model_obj = SingleTastePoisson(
    spike_array=spike_array,
    states=3,  # Number of states to detect
    fit_type='advi'
)

# Generate the PyMC model
model = model_obj.generate_model()
```

**What happens inside the model:**

1. **Priors on changepoint times (tau)**: Where state transitions might occur
2. **Priors on emission rates (lambda)**: Firing rates in each state
3. **Likelihood**: Poisson distribution connecting rates to observed spike counts

The model structure looks like this:

```python
with pm.Model() as model:
    # Prior on changepoint times (sorted)
    tau = pm.Uniform('tau', lower=0, upper=time_bins, shape=states-1)
    tau_sorted = tt.sort(tau)
    
    # Prior on emission rates for each state
    lambda_latent = pm.Exponential('lambda', lam=1.0, shape=states)
    
    # Assign rates to time bins based on changepoints
    # (simplified - actual implementation more complex)
    lambda_t = assign_rates_by_changepoints(tau_sorted, lambda_latent)
    
    # Likelihood: observed spikes follow Poisson distribution
    obs = pm.Poisson('obs', mu=lambda_t, observed=spike_array)
```

For more details on PyMC models, see the [PyMC Model Building Guide](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html).

---

### 2. Inference: Samples and Trace

Once we have a model, we need to perform **inference** to learn about the parameters from the data. PyMC offers two main approaches:

#### Variational Inference (ADVI)

**ADVI** (Automatic Differentiation Variational Inference) is a fast approximate inference method that:

- Finds a simpler distribution that approximates the true posterior
- Is much faster than MCMC but less accurate
- Good for initial exploration or large datasets

```python
from pytau.changepoint_model import advi_fit

# Fit the model using ADVI
model, approx, lambda_stack, tau_samples, data = advi_fit(
    model,
    fit=10000,      # Number of optimization iterations
    samples=5000    # Number of samples to draw from approximation
)
```

**What you get:**
- `approx`: The fitted approximation (a simpler distribution)
- `lambda_stack`: Samples of emission rates (shape: `[samples, states, ...]`)
- `tau_samples`: Samples of changepoint times (shape: `[samples, states-1]`)

#### MCMC Sampling

**MCMC** (Markov Chain Monte Carlo) generates samples by:

- Exploring the parameter space using random walks
- Spending more time in high-probability regions
- Providing high-quality samples but slower than ADVI

```python
from pytau.changepoint_model import mcmc_fit

# Fit the model using MCMC
model, trace, lambda_stack, tau_samples, data = mcmc_fit(
    model,
    samples=1000  # Number of samples per chain
)
```

**What you get:**
- `trace`: An InferenceData object containing all samples
- `lambda_stack`: Samples of emission rates
- `tau_samples`: Samples of changepoint times

For more on inference methods, see:
- [ADVI in PyMC](https://www.pymc.io/projects/docs/en/stable/api/inference/variational.html)
- [MCMC in PyMC](https://www.pymc.io/projects/docs/en/stable/api/inference/sampling.html)

---

### 3. Trace / InferenceData

A **trace** (or **InferenceData** in modern PyMC) is a container for all the samples drawn during inference. Think of it as a table where:

- Each row is a sample from the posterior distribution
- Each column is a parameter in your model

```python
# After fitting with MCMC
print(trace.posterior)
# Output shows available variables: tau, lambda, etc.

# Access specific parameter samples
tau_samples = trace.posterior['tau'].values  # Shape: (chains, draws, states-1)
lambda_samples = trace.posterior['lambda'].values  # Shape: (chains, draws, states, ...)
```

**Key operations with traces:**

```python
import arviz as az

# Summary statistics
summary = az.summary(trace)
print(summary)

# Visualize posterior distributions
az.plot_posterior(trace, var_names=['tau'])

# Check convergence diagnostics
az.plot_trace(trace, var_names=['tau', 'lambda'])
```

**Understanding the samples:**

Each sample represents one plausible set of parameter values given the data. By collecting many samples, we can:

- Estimate parameter means and medians
- Quantify uncertainty with credible intervals
- Visualize the full posterior distribution

For more on working with traces, see the [ArviZ documentation](https://python.arviz.org/en/stable/index.html).

---

### 4. Posterior Predictive

The **posterior predictive distribution** answers: "If my model is correct, what new data would I expect to see?"

It's generated by:

1. Taking parameter samples from the posterior (trace)
2. Simulating new data using those parameters
3. Collecting the simulated datasets

```python
# Generate posterior predictive samples
with model:
    posterior_predictive = pm.sample_posterior_predictive(
        trace,
        var_names=['obs']  # Variable to predict
    )

# Access the predictions
predicted_spikes = posterior_predictive.posterior_predictive['obs'].values
# Shape: (chains, draws, trials, neurons, time)
```

**Why is this useful?**

1. **Model checking**: Compare predicted data to actual data
   - If they look very different, the model may be misspecified
   
2. **Uncertainty quantification**: See the range of possible outcomes
   
3. **Validation**: Check if the model captures important data features

```python
import matplotlib.pyplot as plt

# Compare actual vs predicted data
actual_mean = spike_array.mean(axis=0)  # Average across trials
predicted_mean = predicted_spikes.mean(axis=(0,1))  # Average across chains and draws

plt.figure(figsize=(12, 4))
plt.plot(actual_mean.T, alpha=0.5, label='Actual')
plt.plot(predicted_mean.T, alpha=0.5, linestyle='--', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Spike Count')
plt.legend()
plt.title('Actual vs Posterior Predictive')
plt.show()
```

For more on posterior predictive checks, see the [PyMC Posterior Predictive Guide](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html).

---

## How Everything Fits Together

Here's a complete workflow showing how these concepts connect:

```python
from pytau.changepoint_model import SingleTastePoisson, advi_fit
import arviz as az
import matplotlib.pyplot as plt

# 1. CREATE MODEL
# Define the probabilistic model structure
model_obj = SingleTastePoisson(
    spike_array=spike_array,  # Your data: (trials, neurons, time)
    states=3,
    fit_type='advi'
)
model = model_obj.generate_model()

# 2. PERFORM INFERENCE
# Learn parameter distributions from data
model, approx, lambda_stack, tau_samples, data = advi_fit(
    model,
    fit=10000,
    samples=5000
)

# 3. ANALYZE TRACE
# Examine the posterior distribution
print("Changepoint locations (mean ± std):")
for i in range(tau_samples.shape[1]):
    mean_tau = tau_samples[:, i].mean()
    std_tau = tau_samples[:, i].std()
    print(f"  Tau {i+1}: {mean_tau:.2f} ± {std_tau:.2f}")

# Visualize posterior distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot changepoint distributions
axes[0].hist(tau_samples[:, 0], bins=50, alpha=0.7, label='Tau 1')
axes[0].hist(tau_samples[:, 1], bins=50, alpha=0.7, label='Tau 2')
axes[0].set_xlabel('Time (bins)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Posterior Distribution of Changepoints')
axes[0].legend()

# Plot emission rate distributions
for state in range(lambda_stack.shape[1]):
    axes[1].hist(lambda_stack[:, state, 0, 0], bins=50, alpha=0.5, 
                 label=f'State {state+1}')
axes[1].set_xlabel('Firing Rate (Hz)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Posterior Distribution of Emission Rates')
axes[1].legend()

plt.tight_layout()
plt.show()

# 4. POSTERIOR PREDICTIVE CHECK
# Validate the model by simulating new data
with model:
    # Convert approx to trace for posterior predictive sampling
    idata = approx.sample(draws=1000)
    posterior_predictive = pm.sample_posterior_predictive(
        idata,
        var_names=['obs']
    )

# Compare actual vs predicted
actual_mean = spike_array.mean(axis=0)
predicted_mean = posterior_predictive.posterior_predictive['obs'].values.mean(axis=(0,1))

plt.figure(figsize=(12, 4))
plt.plot(actual_mean[0], label='Actual', linewidth=2)
plt.plot(predicted_mean[0], label='Predicted', linestyle='--', linewidth=2)
plt.xlabel('Time (bins)')
plt.ylabel('Mean Spike Count')
plt.title('Posterior Predictive Check')
plt.legend()
plt.show()
```

---

## Key Takeaways

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| **Model** | Mathematical representation of data generation | Encodes assumptions and structure |
| **Inference** | Process of learning from data | Produces parameter estimates |
| **Samples** | Individual draws from posterior | Represent uncertainty |
| **Trace** | Collection of all samples | Contains full posterior information |
| **Posterior Predictive** | Simulated data from fitted model | Validates model quality |

---

## Common Patterns in PyTau

### Pattern 1: Quick Model Fitting

```python
from pytau.changepoint_model import SingleTastePoisson, advi_fit

# Fit and get results in one go
model_obj = SingleTastePoisson(spike_array=data, states=3, fit_type='advi')
model = model_obj.generate_model()
model, approx, lambda_stack, tau_samples, _ = advi_fit(model, fit=10000, samples=5000)

# Analyze changepoints
mean_changepoints = tau_samples.mean(axis=0)
print(f"Detected changepoints at: {mean_changepoints}")
```

### Pattern 2: Model Comparison

```python
from pytau.changepoint_model import find_best_states

# Automatically find optimal number of states
best_model, model_list, elbo_values = find_best_states(
    data=spike_array,
    model_generator=SingleTastePoisson,
    n_fit=5000,
    n_samples=1000,
    min_states=2,
    max_states=6
)

print(f"Best model has {best_model.kwargs['states']} states")
```

### Pattern 3: High-Quality Sampling

```python
from pytau.changepoint_model import mcmc_fit

# Use MCMC for publication-quality results
model_obj = SingleTastePoisson(spike_array=data, states=3, fit_type='mcmc')
model = model_obj.generate_model()
model, trace, lambda_stack, tau_samples, _ = mcmc_fit(model, samples=2000)

# Check convergence
import arviz as az
print(az.summary(trace, var_names=['tau', 'lambda']))
```

---

## Additional Resources

- [PyMC Documentation](https://www.pymc.io/welcome.html) - Official PyMC guide
- [PyMC Examples](https://www.pymc.io/projects/examples/en/latest/gallery.html) - Gallery of example models
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) - Practical introduction to Bayesian methods
- [ArviZ Documentation](https://python.arviz.org/) - Tools for analyzing Bayesian models
- [PyMC Discourse](https://discourse.pymc.io/) - Community forum for questions

---

## Troubleshooting

### "My samples look weird"

**Check convergence:**
```python
import arviz as az
az.plot_trace(trace)  # Look for "hairy caterpillar" patterns
print(az.summary(trace))  # Check r_hat values (should be ~1.0)
```

### "Inference is too slow"

**Try these approaches:**
1. Use ADVI instead of MCMC for initial exploration
2. Reduce the number of samples
3. Simplify the model (fewer states)
4. Use NumPyro backend for MCMC: `pm.sample(nuts_sampler='numpyro')`

### "Results don't match my expectations"

**Validate your model:**
1. Check posterior predictive samples
2. Visualize the fitted changepoints on your data
3. Try different prior specifications
4. Ensure data preprocessing is correct

---

## Next Steps

Now that you understand the core PyMC concepts, you can:

1. Explore the [API documentation](api.md) for detailed function references
2. Try the example notebooks in `pytau/how_to/notebooks/`
3. Experiment with different model types for your data
4. Customize models for your specific research questions

For questions or issues, please visit the [PyTau GitHub repository](https://github.com/abuzarmahmood/pytau).
