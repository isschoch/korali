# libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

# my code 
from binning_analysis import *
from leapfrog import *


###########################################################################################################
# Hamiltoninian MCMC Start ################################################################################
###########################################################################################################


# potential U = -log[pi(q) * likelihood(q|Data)]
# pi(q): prior density
# likelihood(q|Data): probability that Data was generated with parameters q
# T := 1 assumed


def hamiltonian_mcmc(q_init, data, num_samples, U, grad_U, K, grad_K):
    samples = []
    step_size = 0.001
    num_steps = 10

    q = q_init
    for i in range(num_samples):
        q = hamiltonian_mcmc_sample(
            U, grad_U, K, grad_K, mass_matrix, step_size, num_steps, q)
        samples.append(q)

    return np.array(samples)


# takes potential U and current state q and returns next step based on Hamiltonian dynamcis
def hamiltonian_mcmc_sample(U, grad_U, K, grad_K, mass_matrix, step_size, num_steps, q):
    # 1. Draw momentum variables p from (Gaussian) distribution ###########################################
    p = draw_momentum(mass_matrix)

    # 2. Perform Metropolis update using Hamiltonian dynamics to propose new state in phase space #########
    q_proposed_evolution, p_proposed_evolution = leapfrog(
        q, p, grad_U, grad_K, step_size, num_steps)

    # only interested in last entry but whole evolution captured for possible future visualization
    p_proposed = p_proposed_evolution[:, num_steps-1]
    q_proposed = q_proposed_evolution[:, num_steps-1]

    # negate proposed momentum to make proposal symmetric
    p_proposed = -p_proposed

    # 3. Accept or reject proposed position in phase space ################################################
    q_next = q
    if accept_state_rule(q, q_proposed, p, p_proposed, U, K):
        q_next = q_proposed

    return q_next

# samples momentum (artifically introduced)
def draw_momentum(mass_matrix):
    # here just a Gaussian distribution
    p_proposed = np.random.multivariate_normal(
        np.zeros(shape=(np.shape(mass_matrix)[0])), mass_matrix)

    if p_proposed[1] <= 0.:
        draw_momentum(mass_matrix)

    return p_proposed

# returns bool to indicate if proposed position was accepted
def accept_state_rule(q, q_proposed, p, p_proposed, U, K):
    # accept with probability min[1, exp(-H(q_proposed,p_proposed) + H(q, p))]
    delta_H = K(p_proposed) + U(q_proposed) - (K(p) + U(q))
    # will this really be calculated like this if we have pi(q) = log(...)
    accept_probability = np.exp(-delta_H)

    if accept_probability >= 1.:
        return True
    else:
        if np.random.uniform() <= accept_probability:
            return True
        else:
            return False

    return False


###########################################################################################################
# Hamiltoninian MCMC End ##################################################################################
###########################################################################################################

if __name__ == "__main__":
    verbose = False
    if sys.argv[-1] == "-v":
        verbose = True


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


    num_samples = 100000

    samples = hamiltonian_mcmc(q_init, data, num_samples, U, grad_U, K, grad_K)

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
