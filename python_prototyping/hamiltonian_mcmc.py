# libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

# my code
from statistics import *
from leapfrog import *
from test_hamiltonian_mcmc import *


###########################################################################################################
# Hamiltoninian MCMC Start ################################################################################
###########################################################################################################


# potential U = -log[pi(q) * likelihood(q|Data)]
# pi(q): prior density
# likelihood(q|Data): probability that Data was generated with parameters q
# T := 1 assumed
def hamiltonian_mcmc(num_samples, U, grad_U, step_size, num_steps, q_init=None, dim=None, mass_matrix=None,
                     return_evolution=False, uniform_samples_step_size=False, euclidean_metric=True,
                     num_warmup_samples=1000, warmup_quotient=0.7):
    # assert input
    assert q_init is not None or dim is not None, "Initialize at least one of \"q_init\", \"dim\""
    if q_init is None:
        q_init = np.zeros(dim, dtype=float)
    elif q_init is not None:
        if dim is None:
            dim = len(q_init)
        elif dim is not None:
            assert len(
                q_init) == dim, "dimension of \"q_init\" does not correspond to \"dim\""

    # remove
    metric = mass_matrix

    # initialization
    q = q_init

    if metric is None:
        metric = np.identity(dim, dtype=float)
    metric_inv = np.linalg.inv(metric)

    def K(p):
        return 0.5 * np.matmul(np.matmul(p.T, metric_inv), p)

    def grad_K(p):
        return np.matmul(metric_inv, p)

    # 1. warmup ###########################################################################################
    if euclidean_metric == True:
        metric_inv, q = estimate_metric_inv(
            num_warmup_samples, U, grad_U, step_size, num_steps, metric, metric_inv, q, dim, warmup_quotient)
        metric = np.linalg.inv(metric_inv)

    # sample steps
    samples = []
    num_accepted_proposals = 0
    for i in range(num_samples):
        q, accepted_proposal = hamiltonian_mcmc_sample(
            U, grad_U, K, grad_K, metric, step_size, num_steps, q, return_evolution, uniform_samples_step_size)
        samples.append(q)
        if accepted_proposal:
            num_accepted_proposals += 1

    accept_ratio = num_accepted_proposals / num_samples
    if return_evolution:
        return samples

    return np.array(samples, dtype=float), accept_ratio


def hamiltonian_mcmc_adaptive_time_stepping(num_samples, U, grad_U, num_steps, q_init=None, dim=None, mass_matrix=None, return_evolution=False, uniform_samples_step_size=False, euclidean_metric=True, num_warmup_samples=2000, warmup_quotient=0.7, integration_time=1.0, delta=0.65):
    # assert input
    assert q_init is not None or dim is not None, "Initialize at least one of \"q_init\", \"dim\""
    if q_init is None:
        q_init = np.zeros(dim, dtype=float)
    elif q_init is not None:
        if dim is None:
            dim = len(q_init)
        elif dim is not None:
            assert len(
                q_init) == dim, "dimension of \"q_init\" does not correspond to \"dim\""

    # remove
    metric = mass_matrix

    # initialization
    q = q_init

    if metric is None:
        metric = np.identity(dim, dtype=float)
    metric_inv = np.linalg.inv(metric)

    def K(p):
        return 0.5 * np.matmul(np.matmul(p.T, metric_inv), p)

    def grad_K(p):
        return np.matmul(metric_inv, p)

    step_size = find_reasonable_step_size(q_init, U, grad_U, K, grad_K)

    # set parameters for adaptation of step_size
    mu = np.log(10.0 * step_size)
    H_bar = 0.0
    step_size_bar = 1.0
    gamma = 0.05
    t_0 = 10.0
    kappa = 0.75

    # 1. warmup ###########################################################################################
    if euclidean_metric == True:

        warmup_samples = np.zeros(shape=(num_warmup_samples, dim), dtype=float)

        for m in range(num_warmup_samples):
            u2 = 0.8 + 0.4 * np.random.uniform()
            num_steps = int(
                max(1, np.round(integration_time * u2 / step_size)))

            # 1. Draw momentum variables p from (Gaussian) distribution ###########################################
            p = draw_momentum(metric)

            # 2. Perform Metropolis update using Hamiltonian dynamics to propose new state in phase space #########
            q_proposed_evolution, p_proposed_evolution = leapfrog(
                q, p, grad_U, grad_K, step_size, num_steps)

            # only interested in last entry but whole evolution captured for visualization
            p_proposed = p_proposed_evolution[:, num_steps]
            q_proposed = q_proposed_evolution[:, num_steps]

            # negate proposed momentum to make proposal symmetric
            p_proposed = -p_proposed

            # 3. Accept or reject proposed position in phase space ################################################
            q_next = q

            # accept with probability min[1, exp(-H(q_proposed,p_proposed) + H(q, p))]
            delta_H = K(p_proposed) + U(q_proposed) - (K(p) + U(q))
            # will this really be calculated like this if we have pi(q) = log(...)
            accept_probability = min(1., np.exp(-delta_H))

            accepted_state = False

            if np.random.uniform() <= accept_probability:
                accepted_state = True
            else:
                accepted_state = False

            if accepted_state == True:
                q_next = q_proposed

            q = q_next

            warmup_samples[m, :] = q

            H_bar = (1.0 - 1.0 / (m + 1 + t_0)) * H_bar + \
                (delta - accept_probability) / (m + 1 + t_0)
            step_size = np.exp((mu - np.sqrt(m + 1) / gamma * H_bar))
            step_size_bar = step_size_bar * \
                (step_size / step_size_bar) ** ((m+1) ** (-kappa))

            if m == (int)(warmup_quotient * num_warmup_samples):
                metric_inv = covariance_matrix(warmup_samples, (int)(
                    warmup_quotient * num_warmup_samples))
                metric = np.linalg.inv(metric_inv)
        metric_inv = covariance_matrix(warmup_samples, (int)(
            warmup_quotient * num_warmup_samples))
        metric = np.linalg.inv(metric_inv)

    # sample steps
    samples = np.zeros(shape=(num_samples, dim), dtype=float)
    num_accepted_proposals = 0

    for m in range(num_samples):
        u2 = 0.8 + 0.4 * np.random.uniform()
        num_steps = int(max(1, np.round(integration_time * u2 / step_size)))

        # 1. Draw momentum variables p from (Gaussian) distribution ###########################################
        p = draw_momentum(metric)

        # 2. Perform Metropolis update using Hamiltonian dynamics to propose new state in phase space #########
        q_proposed_evolution, p_proposed_evolution = leapfrog(
            q, p, grad_U, grad_K, step_size, num_steps)

        # only interested in last entry but whole evolution captured for visualization
        p_proposed = p_proposed_evolution[:, num_steps]
        q_proposed = q_proposed_evolution[:, num_steps]

        # negate proposed momentum to make proposal symmetric
        p_proposed = -p_proposed

        # 3. Accept or reject proposed position in phase space ################################################
        q_next = q

        # accept with probability min[1, exp(-H(q_proposed,p_proposed) + H(q, p))]
        delta_H = K(p_proposed) + U(q_proposed) - (K(p) + U(q))
        # will this really be calculated like this if we have pi(q) = log(...)
        accept_probability = min(1., np.exp(-delta_H))

        accepted_state = False

        if np.random.uniform() <= accept_probability:
            accepted_state = True
        else:
            accepted_state = False

        if accepted_state == True:
            q_next = q_proposed
            num_accepted_proposals += 1

        q = q_next

        samples[m, :] = q

        u2 = 0.9 + 0.2 * np.random.uniform()
        step_size = step_size_bar * u2

    accept_ratio = num_accepted_proposals / num_samples

    return samples, accept_ratio

# finds reasonable step_size value according to hmc implementation of previous student


def find_reasonable_step_size(q, U, grad_U, K, grad_K):
    dim = len(q)
    step_size = 1.
    p = draw_momentum(np.identity(dim, dtype=float))

    q_new_evolution, p_new_evolution = leapfrog(
        q, p, grad_U, grad_K, step_size=step_size, num_steps=1)
    q_new = q_new_evolution[:, 1]
    p_new = p_new_evolution[:, 1]

    a = 0.

    if U(q) + K(p) - (U(q_new) + K(p_new)) > np.log(0.5):
        a = 1.
    else:
        a = -1.

    while a * (U(q) + K(p) - (U(q_new) + K(p_new))) > - a * np.log(2):
        step_size = (2 ** a) * step_size
        q_new_evolution, p_new_evolution = leapfrog(
            q, p, grad_U, grad_K, step_size=step_size, num_steps=1)
        q_new = q_new_evolution[:, 1]
        p_new = p_new_evolution[:, 1]

    return step_size


# based on "A Conceptual Introduction to Hamiltonian Monte Carlo" p. 31
# first uses metric given by mass_matrix (default is identity) to sample (warmup_quotient * num_samples)
# samples and uses these to estimate metric_inv. After this estimation is made the remaining
# (1. - warmup_quotient) * num_samples to get an even better estimate of metric_inv


def estimate_metric_inv(num_estimate_samples, U, grad_U, step_size, num_steps, metric, metric_inv, q,
                        dim, warmup_quotient):
    def K(p):
        return 0.5 * np.matmul(np.matmul(p.T, metric_inv), p)

    def grad_K(p):
        return np.matmul(metric_inv, p)

    warmup_samples = np.zeros(shape=(num_estimate_samples, dim), dtype=float)

    for i in range(num_estimate_samples):
        q, accepted_proposal = hamiltonian_mcmc_sample(
            U, grad_U, K, grad_K, metric, step_size, num_steps, q)
        warmup_samples[i, :] = q
        if i == (int)(warmup_quotient * num_estimate_samples):
            metric_inv = covariance_matrix(warmup_samples, (int)(
                warmup_quotient * num_estimate_samples))
            metric = np.linalg.inv(metric_inv)

    metric_inv = covariance_matrix(warmup_samples, num_estimate_samples)

    return metric_inv, q


# returns covariance matrix from samples
def covariance_matrix(samples, num_samples):
    dim = np.shape(samples)[1]
    mu_obs = np.sum(samples, axis=0) / num_samples
    covariance = np.zeros(shape=(dim, dim), dtype=float)

    for i in range(num_samples):
        tmp_mat = np.outer(samples[i, :] - mu_obs, samples[i, :] - mu_obs)
        covariance += tmp_mat

    covariance = covariance / num_samples
    return covariance


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
        step_size_realization = step_size + np.random.uniform(uniform_samples_step_size[0],
                                                              uniform_samples_step_size[1])

    q_proposed_evolution, p_proposed_evolution = leapfrog(
        q, p, grad_U, grad_K, step_size_realization, num_steps)

    # only interested in last entry but whole evolution captured for visualization
    p_proposed = p_proposed_evolution[:, num_steps]
    q_proposed = q_proposed_evolution[:, num_steps]

    # negate proposed momentum to make proposal symmetric
    p_proposed = -p_proposed

    # 3. Accept or reject proposed position in phase space ################################################
    q_next = q
    accepted_proposal = False
    if accept_state_rule(q, q_proposed, p, p_proposed, U, K):
        q_next = q_proposed
        accepted_proposal = True

    # only here for testing and visulization
    if return_evolution:
        return q_next, q_proposed_evolution, p_proposed_evolution

    return q_next, accepted_proposal

# samples momentum (artifically introduced)


def draw_momentum(mass_matrix):
    # here just a Gaussian distribution
    p_proposed = np.random.multivariate_normal(
        np.zeros(shape=(np.shape(mass_matrix)[0])), mass_matrix)

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
