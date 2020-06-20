#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali
e = korali.Experiment()
e["File Output"]["Path"] = "_result_run-mcmc"

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = model
e["Console Output"]["Frequency"] = 500
e["File Output"]["Frequency"] = 500

# Defining problem's variables and their MCMC settings
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "Sampler/MCMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Use Adaptive Sampling"] = True
e["Solver"]["Termination Criteria"]["Max Samples"] = 10000

# Running Korali
k = korali.Engine()
k.run(e)

compareMean(e)
compareStd(e)
