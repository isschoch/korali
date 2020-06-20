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

# Setting up custom likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = llaplaceCustom

# Configuring TMCMC parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 5000

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -20.0
e["Distributions"][0]["Maximum"] = +20.0

# Configuring the problem's variables and their prior distributions
e["Variables"][0]["Name"] = "a"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["File Output"]["Frequency"] = 0

# Running Korali
e["Random Seed"] = 1337
k.run(e)

verifyMean(e["Solver"]["Sample Database"], [4.0], 0.05)
verifyStd(e["Solver"]["Sample Database"], [math.sqrt(2)], 0.05)
