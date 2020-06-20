#!/usr/bin/env/ python

import numpy as np

# fix random seed
np.random.seed(0x007)

# params to generate data
beta0 = 1.0  # distance from shore
alpha0 = 1.0 # position on coastline

# synthetic data
def getReferenceData(N=100):
    angles = np.random.uniform(-np.pi/2,+np.pi/2,N)
    flashes = alpha0 + beta0 * np.tan(angles)
    return flashes

# model for example 1 (known beta)
def model1( s, data ):
    alpha = s["Parameters"][0]
    llk = 0.0

    for i in range(len(data)):
        llk += np.log( 1.0/np.pi * beta0 / ((data[i] - alpha)**2 + beta0**2) )

    s["logLikelihood"] = llk

# model for example 1 (known beta)
def model2( s, data ):
    alpha = s["Parameters"][0]
    beta  = s["Parameters"][1]
    llk = 0.0

    for i in range(len(data)):
        llk += np.log( 1.0/np.pi * beta / ((data[i] - alpha)**2 + beta**2) )

    s["logLikelihood"] = llk
