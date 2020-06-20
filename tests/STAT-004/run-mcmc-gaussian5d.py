#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

lg5 = lambda x: lgaussianxd(x, 5)

# Starting Korali's Engine
import korali
k = korali.Engine()
e = korali.Experiment()

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = lg5
e["Console Output"]["Frequency"] = 5000
e["File Output"]["Frequency"] = 0

# Defining problem's variables and their MCMC settings
for i in range(5):
  e["Variables"][i]["Name"] = "X" + str(i)
  e["Variables"][i]["Initial Mean"] = -1.0
  e["Variables"][i]["Initial Standard Deviation"] = 1.0

# Configuring the MCMC sampler parameters
e["Solver"]["Type"] = "Sampler/MCMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Use Adaptive Sampling"] = False
e["Solver"]["Termination Criteria"]["Max Samples"] = 50000

# Running Korali
e["Random Seed"] = 1337
k.run(e)

verifyMean(e["Solver"]["Sample Database"], [0.0, 0.0, 0.0, 0.0, 0.0], 0.05)
verifyStd(e["Solver"]["Sample Database"], [1.0, 1.0, 1.0, 1.0, 1.0], 0.05)
