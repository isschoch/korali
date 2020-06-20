#!/usr/bin/env python3

# Importing computational model
import sys
sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

lg5 = lambda x: lgaussianxdCustom(x, 5)

# Starting Korali's Engine
import korali
k = korali.Engine()
e = korali.Experiment()

# Setting up custom likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = lg5

# Configuring TMCMC parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 10000

e["Solver"]["Target Coefficient Of Variation"] = 0.4

# Configuring the problem's random distributions
for i in range(5):
  e["Distributions"][i]["Name"] = "Uniform " + str(i)
  e["Distributions"][i]["Type"] = "Univariate/Uniform"
  e["Distributions"][i]["Minimum"] = -5.0
  e["Distributions"][i]["Maximum"] = +5.0

  # Configuring the problem's variables and their prior distributions
  e["Variables"][i]["Name"] = "a"
  e["Variables"][i]["Prior Distribution"] = "Uniform 0"

e["File Output"]["Frequency"] = 0

e["Random Seed"] = 1234

# Running Korali
k.run(e)

verifyMean(e["Solver"]["Sample Database"], [0.0, 0.0, 0.0, 0.0, 0.0], 0.1)
verifyStd(e["Solver"]["Sample Database"], [1.0, 1.0, 1.0, 1.0, 1.0], 0.1)
