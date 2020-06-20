#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

# Starting Korali's Engine
import korali
k = korali.Engine()
e = korali.Experiment()

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = llaplace

# Defining problem's variables and their MCMC settings
e["Variables"][0]["Name"] = "X0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "Sampler/MCMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Use Adaptive Sampling"] = True
e["Solver"]["Termination Criteria"]["Max Samples"] = 100000

e["Console Output"]["Frequency"] = 5000
e["File Output"]["Frequency"] = 0

# Running Korali
e["Random Seed"] = 1227
k.run(e)

verifyMean(e["Solver"]["Sample Database"], [4.0], 0.05)
verifyStd(e["Solver"]["Sample Database"], [math.sqrt(2)], 0.05)
