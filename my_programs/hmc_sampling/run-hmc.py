#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior
# distribution in a bayesian problem where the likelihood
# is provided directly by the computational model.
# In this case, we use the HMC method.

# Importing computational model
import sys
sys.path.append('./_model')
from model import *

# Creating new experiment
import korali
e = korali.Experiment()

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = model

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 50000
e["Solver"]["Termination Criteria"]["Max Samples"] = 50000
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Use Euclidean Metric"] = 1


# Defining problem's variables
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 10.0

e["Variables"][1]["Name"] = "Y"
e["Variables"][1]["Initial Mean"] = 0.0
e["Variables"][1]["Initial Standard Deviation"] = 10.0



# Configuring output settings
e["File Output"]["Frequency"] = 500
e["Console Output"]["Frequency"] = 500
e["Console Output"]["Verbosity"] = "Detailed"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)

