import numpy as np


###########################################################################################################
# Binning analysis Start ##################################################################################
###########################################################################################################

def bin_step(Q_prev):
    N = np.shape(Q_prev)[0]
    M = np.shape(Q_prev)[1]

    Q = np.zeros(shape=(N // 2, M))
    Q = 0.5 * (Q_prev[0:(N-1):2, :] + Q_prev[1:N:2, :])

    return Q


def perform_binning(Q):
    num_levels = (int) (np.log2(np.shape(Q)[0]))
    delta_order_param = np.zeros(shape=(num_levels, np.shape(Q)[1]), dtype=float)

    for l in range(num_levels):
        QAvg = np.average(Q, axis=0)
        M_l = np.shape(Q)[0]

        delta_order_param[l] = np.linalg.norm(
            1. / (M_l * (M_l - 1)) * (Q - QAvg), ord=2, axis=0)

        Q = bin_step(Q)

    return delta_order_param
    
###########################################################################################################
# Binning analysis End ####################################################################################
###########################################################################################################


# based on "Concenptual Introduction to Hamiltonian Monte Carlo" p. 44 section 6.1
def calc_E_BFMI_hat(samples, num_samples):
    mean_obs = np.sum(samples, axis=0) / num_samples

    nominator = np.sum(np.abs(samples[1:num_samples, :] - samples[0:(num_samples-1), :]) ** 2, axis=0)
    denominator = np.sum(np.abs(samples - mean_obs) ** 2, axis=0)

    E_BFMI = nominator / denominator

    return E_BFMI

def calc_ESS_hat(samples, num_samples):
    autocorrelation = perform_binning(samples)

    ESS_hat = num_samples / (1. + 2. * np.sum(autocorrelation, axis=0))

    return ESS_hat