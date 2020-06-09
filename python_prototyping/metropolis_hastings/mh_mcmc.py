import numpy as np
import matplotlib.pyplot as plt
from binning_analysis import *

def prior(q):
    if q[1] <= 0:
        return 0
    return 1

def log_likelihood(q, data):
    n = len(data)
    mu = q[0]
    sigma = q[1]

    return - 0.5 * np.sum(np.abs(data - mu) ** 2) / sigma ** 2 - n * (np.log(sigma) + 0.5 * np.log(2.0 * np.pi))

def accept_state_rule(q, q_proposed, data):
    accept_state = False
    log_accept_probability = log_likelihood(
        q_proposed, data) + np.log(prior(q_proposed)) - (log_likelihood(q, data) + np.log(prior(q)))
    if log_accept_probability > 0:
        accept_state = True
    else:
        u = np.random.uniform()
        if np.log(u) < log_accept_probability:
            accept_state = True
        else:
            accept_state = False
    return accept_state

def propose_transition(q):
    # q_proposed = np.random.normal(q, sigma)

    q_proposed = np.random.multivariate_normal(mean=q, cov=[[1.0, 0.0],
                                                            [0.0, 1.0]])

    if q_proposed[1] < 0:
        q_proposed = propose_transition(q)

    # q_proposed[0] = q[0] #first only implement where sigma is varied
    return q_proposed

def metropolis_hastings(q_init, data, accept_state_rule, propose_transition, num_samples):
    samples = []
    accepted_samples = []
    rejected_samples = []

    q_new = q_init
    for i in range(num_samples):
        q = q_new

        q_proposed = propose_transition(q)

        if accept_state_rule(q, q_proposed, data):
            accepted_samples.append(q_proposed)
            q_new = q_proposed
        else:
            rejected_samples.append(q_proposed)
            q_new = q

        samples.append(q_new)

    return np.array(samples), np.array(accepted_samples), np.array(rejected_samples)

mean = 10
standard_deviation = 3

def mod1(t): return np.random.normal(mean, standard_deviation, t)

# population of size 30000
population_size = 1000
population = mod1(population_size)

# observations (np.random.randint(low, high, size))
num_observations = 500
observation = population[np.random.randint(0, population_size, num_observations)]


# plotting
plt.title("Observed Data")
plt.xlabel("Measurement Value")
plt.ylabel("Number of Occurences")
plt.hist(observation, bins=50)
plt.grid(":")
plt.savefig('population_observations.png', dpi=200)
plt.close()

mu_obs = observation.mean()
print("mu_obs = ", mu_obs)

sigma_obs = np.sqrt(1. / (len(observation) - 1) *
                    np.sum(np.abs(observation - mu_obs)**2))
print("sigma_obs = ", sigma_obs)

q = np.array([mu_obs, sigma_obs])

num_samples = 100000
samples, accepted_samples, rejected_samples = metropolis_hastings(
    q, observation, accept_state_rule, propose_transition, num_samples)

print("samples = ", samples)

plt.title("samples")
plt.xlabel("Iterations")
plt.ylabel("$Parameter Value$")
plt.legend(["$\mu$", "$\sigma$"])
plt.grid(":")
plt.plot(range(num_samples), samples)
plt.savefig("samples.png")
plt.close()


plt.title("Parameter Value $\mu$ Distribution")
plt.xlabel("Parameter Value $\mu$")
plt.ylabel("Number of Occurences")
plt.grid(":")
plt.hist(samples[:, 0], bins=50)
plt.savefig("parameter_distribution_mu.png")
plt.close()

plt.title("Parameter Value $\sigma$ Distribution")
plt.xlabel("Parameter Value $\sigma$")
plt.ylabel("Number of Occurences")
plt.grid(":")
plt.hist(samples[:, 1], bins=50)
plt.savefig("parameter_distribution_sigma.png")
plt.close()

x = samples[:, 0]
y = samples[:, 1]

plt.hist2d(x, y, bins=50, density=False, cmap='plasma')

# Plot a colorbar with label.
cb = plt.colorbar()
cb.set_label('Number of entries')

# Add title and labels to plot.
plt.title('Samples of Posterior $P((\mu, \sigma) \mid \mathcal{D})$')
plt.xlabel('$\mu$')
plt.ylabel('$\sigma$')

# Show the plot.
plt.savefig('heatmap.png')

mu_estimates = np.cumsum(samples[:, 0], axis=0) / np.linspace(1, num_samples + 1, num=num_samples)
print("mu_estimates = ", mu_estimates)

sigma_estimates = np.cumsum(samples[:, 1], axis=0) / np.linspace(1, num_samples + 1, num=num_samples)
print("sigma_estimates = ", sigma_estimates)
