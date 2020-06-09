# libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

# my code
from hamiltonian_mcmc import *

verbose = False
if sys.argv[-1] == "-v":
    verbose = True

# based on https://towardsdatascience.com/from-scratch-bayesian-inference-markov-chain-monte-carlo-and-metropolis-hastings-in-python-ef21a29e25a


def population_test():

    #######################################################################################################
    # Generate Data Start #################################################################################
    #######################################################################################################

    mean = 10
    standard_deviation = 3

    def mod1(t): return np.random.normal(mean, standard_deviation, t)

    # population of size 30000
    population_size = 1000
    population = mod1(population_size)

    # observations (np.random.randint(low, high, size))
    num_data_points = 500
    data = population[np.random.randint(0, population_size, num_data_points)]
    mu_obs = data.mean()
    sigma_obs = np.sqrt(1. / (len(data) - 1) *
                        np.sum(np.abs(data - mu_obs)**2))

    if verbose:
        print("mu_obs = ", mu_obs)
        print("sigma_obs = ", sigma_obs)

    q_init = np.array([mu_obs, sigma_obs])

    #######################################################################################################
    # Generate Data End ###################################################################################
    #######################################################################################################

    #######################################################################################################
    # Energy definitions Start ############################################################################
    #######################################################################################################

    def prior(q):
        if q[1] <= 0:
            return 0
        return 1

    def log_likelihood(q, data):
        n = len(data)
        mu = q[0]
        sigma = q[1]

        return - 0.5 * np.sum(np.abs(data - mu) ** 2) / sigma ** 2 - n * (np.log(sigma) + 0.5 * np.log(2.0 * np.pi))

    mass = 0.01 * np.ones(shape=(2), dtype=float)
    mass_matrix = np.diag(mass)

    def U(q): return - (log_likelihood(q, data) + np.log(prior(q)))

    def grad_U(q):
        mu = q[0]
        sigma = q[1]
        n = len(data)

        gradient = np.zeros(shape=(len(q_init)), dtype=float)
        gradient[0] = np.sum(data - mu) / sigma ** 2
        gradient[1] = - n / sigma + np.sum(np.abs(data - mu) ** 2) / sigma ** 3
        return gradient

    def K(p):
        return np.matmul(np.matmul(p.T, np.linalg.inv(mass_matrix)), p) / 2.

    def grad_K(p):
        return np.matmul(np.linalg.inv(mass_matrix), p)

    #######################################################################################################
    # Energy definitions End ##############################################################################
    #######################################################################################################

    #######################################################################################################
    # Run Hamiltonian MCMC Start ##########################################################################
    #######################################################################################################

    num_samples = 1000
    num_warmup_samples = 0

    samples = hamiltonian_mcmc(
        q_init, num_samples, num_warmup_samples, U, grad_U, K, grad_K, mass_matrix)

    if verbose:
        print("samples = ", samples)

    #######################################################################################################
    # Run Hamiltonian MCMC End ############################################################################
    #######################################################################################################

    #######################################################################################################
    # Plotting Start ######################################################################################
    #######################################################################################################

    if verbose:
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

        mu_estimates = np.cumsum(samples[:, 0], axis=0) / \
            np.linspace(1, num_samples + 1, num=num_samples)
        print("mu_estimates = ", mu_estimates)

        sigma_estimates = np.cumsum(
            samples[:, 1], axis=0) / np.linspace(1, num_samples + 1, num=num_samples)
        print("sigma_estimates = ", sigma_estimates)

    #######################################################################################################
    # Plotting End ########################################################################################
    #######################################################################################################

# based on section 3.3 "Illustrations of HMC and its benefits" from "MCMC Using Hamiltonian Dynamics" on page 15


def gaussian_2d_test():
    mass_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    covariance_matrix = np.array([[1.0, 0.95], [0.95, 1.0]], dtype=float)
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)

    if verbose:
        print("mass_matrix = ", mass_matrix)
        print("covariance_matrix = ", covariance_matrix)
        print("covariance_matrix_inv = ", covariance_matrix_inv)

    #######################################################################################################
    # Energy definitions Start ############################################################################
    #######################################################################################################

    def U(q):
        return np.matmul(np.matmul(q.T, covariance_matrix_inv), q) / 2.

    def grad_U(q):
        return np.matmul(covariance_matrix_inv, q)

    def K(p):
        return np.dot(p, p) / 2.

    def grad_K(p):
        return p

    #######################################################################################################
    # Energy definitions End ##############################################################################
    #######################################################################################################

    #######################################################################################################
    # Run Hamiltonian MCMC Start ##########################################################################
    #######################################################################################################

    q_init = np.array([-1.50, -1.55]).T

    num_samples = 1
    num_warmup_samples = 0
    step_size = 0.25
    num_steps = 25
    return_evolution = True

    samples = hamiltonian_mcmc(q_init, num_samples, num_warmup_samples,
                               U, grad_U, K, grad_K, mass_matrix, step_size, num_steps, return_evolution)

    # print("samples = ", samples)
    # print("samples[0] = ", samples[0][1])
    # print("samples[0, 1] = ", samples[0, 1])
    q_evolution = np.array(samples[0][1])
    p_evolution = np.array(samples[0][2])

    Hamiltonian_value = np.zeros(shape=(num_steps))
    for i in range(num_steps):
        Hamiltonian_value[i] = K(p_evolution[:, i]) + U(q_evolution[:, i])

    if verbose:
        print("q_evolution = ", q_evolution)
        print("p_evolution = ", p_evolution)
        print("Hamiltonian_value = ", Hamiltonian_value)

        plt.title("Value of Hamiltonian")
        plt.xlabel("Iterations")
        plt.ylabel("$H(q, p)$")
        plt.grid(":")
        plt.plot(range(num_steps), Hamiltonian_value, "r-")
        plt.savefig("value_of_hamiltonian.png")
        plt.close()


    #######################################################################################################
    # Run Hamiltonian MCMC End ############################################################################
    #######################################################################################################
