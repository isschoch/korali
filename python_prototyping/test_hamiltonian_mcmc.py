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
    population_size = 100000
    population = mod1(population_size)

    # observations (np.random.randint(low, high, size))
    num_data_points = 5000
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

    mass = np.ones(shape=(2), dtype=float)
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

    #######################################################################################################
    # Energy definitions End ##############################################################################
    #######################################################################################################

    #######################################################################################################
    # Run Hamiltonian MCMC Start ##########################################################################
    #######################################################################################################

    num_samples = 1000

    samples = hamiltonian_mcmc(num_samples, U, grad_U, step_size=0.001, num_steps=10,
                               mass_matrix=mass_matrix, q_init=q_init)

    if verbose:
        print("samples = ", samples)

    #######################################################################################################
    # Run Hamiltonian MCMC End ############################################################################
    #######################################################################################################

    #######################################################################################################
    # Plotting Start ######################################################################################
    #######################################################################################################

    mu_estimates = np.cumsum(samples[:, 0], axis=0) / \
        np.linspace(1, num_samples + 1, num=num_samples)

    sigma_estimates = np.cumsum(
        samples[:, 1], axis=0) / np.linspace(1, num_samples + 1, num=num_samples)
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
        plt.close()

        print("mu_estimates = ", mu_estimates)
        print("sigma_estimates = ", sigma_estimates)

    #######################################################################################################
    # Plotting End ########################################################################################
    #######################################################################################################

    #######################################################################################################
    # Check test Start ####################################################################################
    #######################################################################################################

    final_mu_estimate = mu_estimates[-1]
    final_sigma_estimate = sigma_estimates[-1]

    tolerance = 0.2

    test_passed = False
    if np.abs(mu_obs - final_mu_estimate) + np.abs(sigma_obs - final_sigma_estimate) < tolerance:
        test_passed = True

    print("Population test passed? (stochastic): ", test_passed)

    #######################################################################################################
    # Check test End ######################################################################################
    #######################################################################################################


# based on section 3.3 "Illustrations of HMC and its benefits" from "MCMC Using Hamiltonian Dynamics"
# on page 15 (Figure 3)


def gaussian_2d_trajectory_test():
    mass_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    mass_matrix_inv = np.linalg.inv(mass_matrix)

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
        return 0.5 * np.matmul(np.matmul(p.T, mass_matrix_inv), p)

    #######################################################################################################
    # Energy definitions End ##############################################################################
    #######################################################################################################

    #######################################################################################################
    # Run Hamiltonian MCMC Start ##########################################################################
    #######################################################################################################

    q_init = np.array([-1.50, -1.55]).T

    num_samples = 1
    step_size = 0.25
    num_steps = 25

    samples = hamiltonian_mcmc(num_samples, U, grad_U, step_size, num_steps, q_init=q_init,
                               mass_matrix=mass_matrix, return_evolution=True, euclidean_metric=False)

    q_evolution = np.array(samples[0][1])
    p_evolution = np.array(samples[0][2])

    Hamiltonian_value = np.zeros(shape=(num_steps))
    for i in range(num_steps):
        Hamiltonian_value[i] = K(p_evolution[:, i]) + U(q_evolution[:, i])

    #######################################################################################################
    # Run Hamiltonian MCMC End ############################################################################
    #######################################################################################################

    #######################################################################################################
    # Plotting Start ######################################################################################
    #######################################################################################################

    if verbose:
        print("q_evolution = ", q_evolution)
        print("p_evolution = ", p_evolution)
        print("Hamiltonian_value = ", Hamiltonian_value)

        plt.title("Position coordinates $q$")
        plt.xlabel("$q_{1}$")
        plt.ylabel("$q_{2}$")
        plt.axes().set_aspect("equal")
        plt.grid(":")
        plt.plot(q_evolution[0, :], q_evolution[1, :], "ro-")
        plt.savefig("position_coordinates.png", dpi=200)
        plt.close()

        plt.title("Momentum coordinates $p$")
        plt.xlabel("$p_{1}$")
        plt.ylabel("$p_{2}$")
        plt.axes().set_aspect("equal")
        plt.grid(":")
        plt.plot(p_evolution[0, :], p_evolution[1, :], "ro-")
        plt.savefig("momentum_coordinates.png", dpi=200)
        plt.close()

        plt.title("Value of Hamiltonian")
        plt.xlabel("Iterations")
        plt.ylabel("$H(q, p)$")
        plt.grid(":")
        # plt.axes().set_aspect("equal")
        plt.plot(range(num_steps), Hamiltonian_value, "ro-")
        plt.savefig("value_of_hamiltonian.png", dpi=200)
        plt.close()

    #######################################################################################################
    # Plotting End ########################################################################################
    #######################################################################################################

    #######################################################################################################
    # Check test Start ####################################################################################
    #######################################################################################################

    # These are just previously calculated values which replicate values of paper upon visual inspection
    q_evolution_reference = np.array([
        [-1.5, -1.7411859, -1.63349564, -1.22319936, -0.87907908, -0.83606942, -0.90784623, -0.72742684,
         -0.21478501, 0.31332301, 0.52891769, 0.49212461, 0.56545281, 0.95051365, 1.42660692, 1.61824267,
         1.45746437, 1.26329492, 1.33917439, 1.59171351, 1.64738346, 1.32128121, 0.85195434, 0.60863579,
         0.63667145, 0.60913276],
        [-1.55, -1.2599359, -1.22255814, -1.39624624, -1.41980174, -1.06856454, -0.54149491, -0.22016824,
         -0.20069237, -0.18336611, 0.14230821, 0.69885663, 1.1071113, 1.1500255, 1.03458231, 1.12471236,
         1.47934124, 1.77323288, 1.69975099, 1.352208, 1.10717772, 1.15563242, 1.26792349, 1.08626143,
         0.57892154, 0.08819468]])

    p_evolution_reference = np.array([
        [-1.00000000e+00, -2.66991289e-01, 1.03597307e+00, 1.50883313e+00, 7.74259882e-01, -5.75342968e-02,
         2.17285156e-01, 1.38612244e+00, 2.08149971e+00, 1.48740539e+00, 3.57603194e-01, 7.30702353e-02,
         9.16778076e-01, 1.72230822e+00, 1.33545804e+00, 6.17149008e-02, -
         7.09895502e-01, -2.36579951e-01,
         6.56837188e-01, 6.16418132e-01, -5.40864604e-01, -
         1.59085823e+00, -1.42529084e+00, -4.30565788e-01,
         9.93937798e-04, -7.83677599e-01],
        [1.00000000e+00, 6.54883711e-01, -2.72620677e-01, -3.94487183e-01, 6.55363398e-01, 1.75661365e+00,
         1.69679260e+00, 6.81605076e-01, 7.36042432e-02, 6.86001157e-01, 1.76444549e+00, 1.92960619e+00,
         9.02337752e-01, -1.45057983e-01, -
         5.06262888e-02, 8.89517856e-01, 1.29704104e+00, 4.40819500e-01,
         -8.42049765e-01, -1.18514653e+00, -
         3.93151150e-01, 3.21491526e-01, -1.38741980e-01, -1.37800390,
         -1.99613351, -1.33408507e+00]])

    Hamiltonian_value_reference = np.array(
        [2.20512821, 2.56279556, 2.46390404, 2.24550376, 2.65393999, 2.27985159, 2.40353181, 2.59463697,
         2.19215329, 2.57715188, 2.4256705, 2.2599011, 2.65296745, 2.25839357, 2.43801851, 2.57997623,
         2.2012736, 2.6074451, 2.40430188, 2.2901983, 2.65543251, 2.24084239, 2.46664202, 2.55106397,
         2.19839464])

    test_passed = True
    tolerance = 1.e-6

    if np.linalg.norm(q_evolution - q_evolution_reference) > tolerance:
        test_passed = False

    if np.linalg.norm(p_evolution - p_evolution_reference) > tolerance:
        test_passed = False

    if np.linalg.norm(Hamiltonian_value - Hamiltonian_value_reference) > tolerance:
        test_passed = False

    print("gaussian_2d_trajectory_test passed?: ", test_passed)

    #######################################################################################################
    # Check test End ######################################################################################
    #######################################################################################################

# based on section 3.3 "Illustrations of HMC and its benefits" from "MCMC Using Hamiltonian Dynamics"
# on page 17 (Figure 4 & 5)


def gaussian_2d_sampling_test():
    mass_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    covariance_matrix = np.array([[1.0, 0.98], [0.98, 1.0]], dtype=float)
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

    q_start = np.array([-1.50, -1.55]).T

    num_samples = 200
    step_size = 0.18
    num_steps = 20

    samples = hamiltonian_mcmc(num_samples, U, grad_U, step_size, num_steps, mass_matrix=mass_matrix,
                               q_init=q_start)

    #######################################################################################################
    # Run Hamiltonian MCMC End ############################################################################
    #######################################################################################################

    #######################################################################################################
    # Plotting Start ######################################################################################
    #######################################################################################################

    if verbose:
        print("samples = ", samples)

        plt.title("Position coordinates $q$ (Samples)")
        plt.xlabel("$q_{1}$")
        plt.ylabel("$q_{2}$")
        plt.axes().set_aspect("equal")
        plt.grid(":")
        plt.plot(samples[0:20, 0], samples[0:20, 1], "ro-")
        plt.savefig("samples.png", dpi=200)
        plt.close()

        plt.title("First Position Coordinate $q_{1}$")
        plt.xlabel("Iterations")
        plt.ylabel("$q_{1}$")
        plt.grid(":")
        # plt.axes().set_aspect("equal")
        plt.plot(range(num_samples), samples[:, 0], "ro")
        plt.savefig("first_position_coordinate.png", dpi=200)
        plt.close()

    #######################################################################################################
    # Plotting End ########################################################################################
    #######################################################################################################

    #######################################################################################################
    # Check test Start ####################################################################################
    #######################################################################################################

    # Due to stochastisity of test (momentum is drawn randomly from Gaussian distribution) one has to
    # compare plots with paper "MCMC Using Hamiltonian Dynamcis" on page 17
    print("gaussian_2d_sampling_test passed?: Check plots")

    #######################################################################################################
    # Check test End ######################################################################################
    #######################################################################################################


def gaussian_100d_sampling_test():
    dim = 100
    mass_matrix = np.identity(dim, dtype=float)

    # covariance_matrix = np.zeros(shape(dim, dim), dtype=float)
    standard_deviation = np.linspace(0.01, 1.00, dim, dtype=float)
    covariance_diag = np.multiply(standard_deviation, standard_deviation)
    covariance_matrix = np.diag(covariance_diag)
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)

    if verbose:
        print("mass_matrix_diag = ", np.diag(mass_matrix))
        print("covariance_diag = ", covariance_diag)
        print("covariance_diag_inv = ", 1. / covariance_diag)

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

    q_init = np.zeros(dim, dtype=float)

    num_samples = 1000
    step_size = 0.013
    num_steps = 150

    samples = hamiltonian_mcmc(num_samples, U, grad_U, step_size, num_steps,
                               mass_matrix=mass_matrix, return_evolution=False,
                               uniform_samples_step_size=[-0.20 *
                                                          step_size, 0.20 * step_size],
                               q_init=q_init)

    #######################################################################################################
    # Run Hamiltonian MCMC End ############################################################################
    #######################################################################################################

    #######################################################################################################
    # Plotting Start ######################################################################################
    #######################################################################################################

    estimated_mean = np.sum(samples, axis=0) / num_samples
    estimated_standard_deviation = np.zeros(dim, dtype=float)

    for i in range(dim):
        estimated_standard_deviation[i] = np.sqrt(1. / (num_samples - 1.) * np.sum(
            np.abs(samples[:, i] - estimated_mean[i]) ** 2))

    if verbose:
        print("samples = ", samples)
        print("estimated_mean = ", estimated_mean)
        print("estimated_standard_deviation = ", estimated_standard_deviation)

        plt.title("Estimate of Expectation Value $\mu$")
        plt.xlabel("Standard Deviation $\sigma$ of Coordinate")
        plt.ylabel("Estimated Expectation Value")
        plt.ylim(-0.7, 0.7)
        # plt.axes().set_aspect("equal")
        plt.grid(":")
        plt.plot(standard_deviation, np.zeros(dim, dtype=float), "b-")
        plt.plot(standard_deviation, estimated_mean, "r.")
        plt.legend(["True Value", "Measured"])
        plt.savefig("estimated_mean.png", dpi=200)
        plt.close()

        plt.title("Estimate of Standard Deviation $\sigma$")
        plt.xlabel("Standard Deviation $\sigma$ of Coordinate")
        plt.ylabel("Estimated Standard Deviation")
        plt.ylim(0.0, 1.2)
        plt.axes().set_aspect("equal")
        plt.grid(":")
        plt.plot(standard_deviation, standard_deviation, "b-")
        plt.plot(standard_deviation, estimated_standard_deviation, "r.")
        plt.legend(["True Value", "Measured"])
        plt.savefig("estimated_standard_deviation.png", dpi=200)
        plt.close()

    #######################################################################################################
    # Plotting End ########################################################################################
    #######################################################################################################

    #######################################################################################################
    # Check test Start ####################################################################################
    #######################################################################################################

    test_passed = True
    tolerance = 5.0e-1
    if np.linalg.norm(estimated_mean - np.zeros(dim, dtype=float)) > tolerance:
        test_passed = False

    if np.linalg.norm(estimated_standard_deviation - standard_deviation) > tolerance:
        test_passed = False

    print("gaussian_100d_sampling_test passed? (stochastic): ", test_passed)

    #######################################################################################################
    # Check test End ######################################################################################
    #######################################################################################################
