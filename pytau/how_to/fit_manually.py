## Fit a model to data manually

import numpy as np
import sys
sys.path.append('/media/bigdata/projects/pytau')
import pytau

# Load spikes
spikes = ...

# Reshape + Bin Spikes
binned_spike_array = ... # shape = (n_trials, n_neurons, n_bins)

# Create and fit model
n_fit = 40000
n_samples = 20000
model = pytau.changepoint_model.single_taste_poisson(spike_array, n_states)
with model:
    inference = pm.ADVI('full-rank') # ADVI is preferred over MCMC due to speed
    approx = pm.fit(n=n_fit, method=inference)
    trace = approx.sample(draws=samples)

# Plot ELBO over iterations, should be flat by the end
fig,ax = plt.subplots(1,2,figsize=(15,5))
ax[0].plot(-approx.hist, alpha=.3)
ax[0].set_ylabel('ELBO')
ax[0].set_xlabel('iteration');

ind = int(n_fit - n_fit*0.05)
last_iters = approx.hist[ind:]
ax[1].plot(-last_iters, alpha=.3)
x = np.arange(len(last_iters))
binned_iters = np.reshape(last_iters, (-1, 100)).mean(axis=-1)
binned_x = x[::100]
ax[1].plot(binned_x, -binned_iters)
ax[1].set_title('Final 5% of iterations')
ax[1].set_ylabel('ELBO')
ax[1].set_xlabel('iteration');

# Extract changepoint values
tau_stack = trace['tau']
int_tau = np.vectorize(np.int)(tau_stack)
mode_tau = np.squeeze(stats.mode(int_tau,axis=0)[0])
