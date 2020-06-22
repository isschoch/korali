# libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

# my code
from statistics import *
from leapfrog import *
from test_hamiltonian_mcmc import *

def nuts(num_samples, U, grad_U, q_init=None, dim=None, metric=None, euclidean_metric=True, num_warmup_samples=2000, warmup_quotient=0.7, integration_time=1.0, delta=0.65):
    if metric is None:
        metric = np.identity(dim, dtype=float)
    metric_inv = np.linalg.inv(metric)

    def K(p):
        return 0.5 * np.matmul(np.matmul(p.T, metric_inv), p)

    def grad_K(p):
        return np.matmul(metric_inv, p)
    
    def H(q, p):
        return U(q) + K(p)

    step_size = find_reasonable_step_size(q_init, U, grad_U, K, grad_K)
    step_size_bar = 1.0

    # set parameters for adaptation of step_size
    mu = np.log(10.0 * step_size)
    H_bar = 0.0
    step_size_bar = 1.0
    gamma = 0.05
    t_0 = 10.0
    kappa = 0.75

    warmup_samples = np.zeros(shape=(num_warmup_samples, dim), dtype=float)
    warmup_samples[0, :] = q_init
    for m in range(1, num_warmup_samples):
        p_0 = draw_momentum(metric)
        
        u = np.random.uniform(0.0, np.exp(-H(warmup_samples[m-1, :], p_0)))
        
        q_minus = warmup_samples[m-1, :]
        q_plus = warmup_samples[m-1, :]
        p_minus = p_0
        p_plus = p_0
        j = 0
        warmup_samples[m, :] = warmup_samples[m-1, :]
        n = 1
        s = 1
        step_size_old = step_size

        while s == 1:
            v = 2. * (np.random.randint(0, 2) - 0.5)

            if v == -1:
                q_minus, p_minus, _, _, q_prime, n_prime, s_prime, alpha, n_alpha = build_tree(q_minus, p_minus, u, v, j, step_size_old, warmup_samples[m-1, :], p_0, U, grad_U, K, grad_K)
            elif v == 1:
                _, _, q_plus, p_plus, q_prime, n_prime, s_prime, alpha, n_alpha = build_tree(q_plus, p_plus, u, v, j, step_size_old, warmup_samples[m-1, :], p_0, U, grad_U, K, grad_K)
            else:
                print("ERROR: v is neither -1 nor 1")
            
            if s_prime == 1:
                accept_probability = min(1.0, n_prime / n)
                if np.random.uniform() <= accept_probability:
                    warmup_samples[m, :] = q_prime

            n = n + n_prime
            
            if np.dot(q_plus - q_minus, p_minus) >= 0 and np.dot(q_plus - q_minus, p_plus) >= 0:
                s = s_prime
            else:
                s = 0
            
            j = j + 1

        H_bar = (1.0 - 1.0 / (m + 1 + t_0)) * H_bar + (delta - alpha) / (m + 1 + t_0)
        step_size = np.exp((mu - np.sqrt(m + 1) / gamma * H_bar))
        step_size_bar = step_size_bar * (step_size / step_size_bar) ** ((m+1) ** (-kappa))
        

    step_size = step_size_bar

    samples = np.zeros(shape=(num_samples, dim), dtype=float)
    samples[0, :] = warmup_samples[num_warmup_samples - 1, :]
    for m in range(1, num_samples):
        p_0 = draw_momentum(metric)
        
        u = np.random.uniform(0.0, np.exp(-H(samples[m-1, :], p_0)))
        
        q_minus = samples[m-1, :]
        q_plus = samples[m-1, :]
        p_minus = p_0
        p_plus = p_0
        j = 0
        samples[m, :] = samples[m-1, :]
        n = 1
        s = 1
        step_size_old = step_size

        while s == 1:
            v = 2. * (np.random.randint(0, 2) - 0.5)

            if v == -1:
                q_minus, p_minus, _, _, q_prime, n_prime, s_prime, alpha, n_alpha = build_tree(q_minus, p_minus, u, v, j, step_size_old, samples[m-1, :], p_0, U, grad_U, K, grad_K)
            elif v == 1:
                _, _, q_plus, p_plus, q_prime, n_prime, s_prime, alpha, n_alpha = build_tree(q_plus, p_plus, u, v, j, step_size_old, samples[m-1, :], p_0, U, grad_U, K, grad_K)
            else:
                print("ERROR: v is neither -1 nor 1")
            
            if s_prime == 1:
                accept_probability = min(1.0, n_prime / n)
                if np.random.uniform() <= accept_probability:
                    samples[m, :] = q_prime

            n = n + n_prime
            
            if np.dot(q_plus - q_minus, p_minus) >= 0 and np.dot(q_plus - q_minus, p_plus) >= 0:
                s = s_prime
            else:
                s = 0
            
            j = j + 1



    return samples



# TODO: Test build_tree
# builds tree for NUTS algorithm
def build_tree(q, p, u, v, j, step_size, q_0, p_0, U, grad_U, K, grad_K):
    delta_max = 100
    if j == 0:
        ################################## FIRST OPTION ############################################
        
        # q_minus, p_minus = leapfrog_step(q, p, grad_U, grad_K, v * step_size, num_steps=1)
        # H_prime = U(q_minus) + K(p_minus)
        
        # # calc n_prime
        # n_prime = -1
        # # TODO: Why is this different from formula on page 1369?
        # if u <= np.exp(H_0 - H_prime):
        #     n_prime = 1
        # else:
        #     n_prime = 0

        # # calc s_prime
        # s_prime = -1
        # # TODO: Why is this different from formula on page 1369?
        # if - H_prime > - H_0 - delta_max:
        #     s_prime = 1
        # else:
        #     s_prime
        
        # accept_probability = min(1., np.exp(H_0 - H_prime))
        # n_alpha_prime = 1
        # return q_minus, p_minus, q_minus, p_minus, q_minus, n_prime, s_prime, accept_probability, n_alpha_prime


        ################################## SECOND OPTION ###########################################
        q_prime, p_prime = leapfrog_step(q, p, grad_U, grad_K, v * step_size)
        H_prime = U(q_prime) + K(p_prime)
        H_0     = U(q_0) + K(p_0)

        n_prime = -1

        # TODO: Why H_0 here contrary to algorithm? If I leave it away there is a possibility of dividing by zero when cheking to set q_prime 
        if u <= np.exp( -H_prime):
            n_prime = 1
        else:
            n_prime = 0
        
        s_prime = -1
        if u <= np.exp(delta_max - H_prime):
            s_prime = 1
        else:
            s_prime = 0
        
        accept_probability = min(1., np.exp(H_0 - H_prime))
        return q_prime, p_prime, q_prime, p_prime, q_prime, n_prime, s_prime, accept_probability, 1
    else:

        # TODO: Check argument order and naming
        q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = build_tree(q, p, u, v, j-1, step_size, q_0, p_0, U, grad_U, K, grad_K)

        if s_prime == 1:
            if v == -1:
                q_minus, p_minus, _, _, q_double_prime, n_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime = build_tree(q_minus, p_minus, u, v, j-1, step_size, q_0, p_0, U, grad_U, K, grad_K)
            else:
                _, _, q_plus, p_plus, q_double_prime, n_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime = build_tree(q_plus, p_plus, u, v, j-1, step_size, q_0, p_0, U, grad_U, K, grad_K)
            
            if np.random.uniform() < n_double_prime / (n_prime + n_double_prime):
                q_prime = q_double_prime
            
            alpha_prime = alpha_prime + alpha_double_prime
            n_alpha_prime = n_alpha_prime + n_alpha_double_prime

            if np.dot(q_plus - q_minus, p_minus) >= 0 and np.dot(q_plus - q_minus, p_plus) >= 0:
                s_prime = s_double_prime
            else:
                s_prime = 0
            
            n_prime = n_prime + n_double_prime
        return q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime, alpha_prime, n_alpha_prime


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



def draw_momentum(mass_matrix):
    # here just a Gaussian distribution
    p_proposed = np.random.multivariate_normal(
        np.zeros(shape=(np.shape(mass_matrix)[0])), mass_matrix)

    return p_proposed



if __name__ == "__main__":
    q = np.ones(shape=(2), dtype=float)
    p = 0.5 * np.ones(shape=(2), dtype=float)
    u = 0.5
    v = 1
    j = 1
    step_size = 0.001
    # q_0 = np.zeros(shape=(2), dtype=float)
    # p_0 = np.zeros(shape=(2), dtype=float)   
    q_0 = q
    p_0 = p


    mass_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    covariance_matrix = np.array([[1.0, 0.98], [0.98, 1.0]], dtype=float)
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)

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

    num_samples = 10
    num_warmup_samples = 10
    q_init = np.ones(shape=(2), dtype=float)
    dim = np.shape(q_init)[0]
    samples = nuts(num_samples, U, grad_U, q_init, dim, mass_matrix, num_warmup_samples=num_warmup_samples)

    if verbose:
        print("samples = ", samples)
        q_estimate = np.average(samples, axis=0)
        print("q_estimate = ", q_estimate)
