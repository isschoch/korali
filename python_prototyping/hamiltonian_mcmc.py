# libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

# my code
from binning_analysis import *
from leapfrog import *
from test_hamiltonian_mcmc import *


###########################################################################################################
# Hamiltoninian MCMC Start ################################################################################
###########################################################################################################


# potential U = -log[pi(q) * likelihood(q|Data)]
# pi(q): prior density
# likelihood(q|Data): probability that Data was generated with parameters q
# T := 1 assumed


def hamiltonian_mcmc(q_init, num_samples, num_warmup_samples, U, grad_U, K, grad_K, mass_matrix,
                     step_size=0.001, num_steps=10, return_evolution=False,
                     uniform_samples_step_size=False):
    # initialization
    samples = []
    q = q_init

    # warmup
    for i in range(num_warmup_samples):
        q = hamiltonian_mcmc_sample(
            U, grad_U, K, grad_K, mass_matrix, step_size, num_steps, q, return_evolution,
            uniform_samples_step_size)

    # sample steps
    for i in range(num_samples):
        q = hamiltonian_mcmc_sample(
            U, grad_U, K, grad_K, mass_matrix, step_size, num_steps, q, return_evolution,
            uniform_samples_step_size)
        samples.append(q)

    if return_evolution:
        return samples

    return np.array(samples)


# takes potential U and current state q and returns next step based on Hamiltonian dynamcis
def hamiltonian_mcmc_sample(U, grad_U, K, grad_K, mass_matrix, step_size, num_steps, q,
                            return_evolution=False, uniform_samples_step_size=False):
    # 1. Draw momentum variables p from (Gaussian) distribution ###########################################
    p = draw_momentum(mass_matrix)

    if return_evolution == True:
        p = np.array([-1.0, 1.0], dtype=float).T

    # 2. Perform Metropolis update using Hamiltonian dynamics to propose new state in phase space #########
    if uniform_samples_step_size == False:
        step_size_realization = step_size
    else:
        step_size_realization = step_size + np.random.uniform(uniform_samples_step_size[0], uniform_samples_step_size[1])

    q_proposed_evolution, p_proposed_evolution = leapfrog(
        q, p, grad_U, grad_K, step_size_realization, num_steps)

    # only interested in last entry but whole evolution captured for possible future visualization
    p_proposed = p_proposed_evolution[:, num_steps-1]
    q_proposed = q_proposed_evolution[:, num_steps-1]

    # negate proposed momentum to make proposal symmetric
    p_proposed = -p_proposed

    # 3. Accept or reject proposed position in phase space ################################################
    q_next = q
    if accept_state_rule(q, q_proposed, p, p_proposed, U, K):
        q_next = q_proposed

    # only here for testing and visulization
    if return_evolution:
        return q_next, q_proposed_evolution, p_proposed_evolution

    return q_next

# samples momentum (artifically introduced)


def draw_momentum(mass_matrix):
    # here just a Gaussian distribution
    p_proposed = np.random.multivariate_normal(
        np.zeros(shape=(np.shape(mass_matrix)[0])), mass_matrix)

    # if p_proposed[1] <= 0.:
    #     draw_momentum(mass_matrix)

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

###########################################################################################################
# Parameters Start ########################################################################################
###########################################################################################################

# TODO

###########################################################################################################
# Parameters End ##########################################################################################
###########################################################################################################


if __name__ == "__main__":
    population_test()
    gaussian_2d_trajectory_test()
    gaussian_2d_sampling_test()
    gaussian_100d_sampling_test()
