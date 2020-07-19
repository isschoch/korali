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

e["File Output"]["Frequency"] = 0
e["Console Output"]["Frequency"] = 5000

# Selecting problem and solver types.
e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = lgaussian

# Defining problem's variables and their HMC settings
e["Variables"][0]["Name"] = "X0"
e["Variables"][0]["Initial Mean"] = 0.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 100
e["Solver"]["Termination Criteria"]["Max Samples"] = 100000

# HMC specific parameters
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.05
e["Solver"]["Use Euclidean Metric"] = 1
e["Solver"]["Use Adaptive Step Size"] = 1
e["Solver"]["Target Integration Time"] = 0.5
e["Solver"]["Desired Average Acceptance Rate"] = 0.80
e["Solver"]["Use NUTS"] = 0

# Running Korali
e["Random Seed"] = 1227
k.run(e)

verifyMean(e["Solver"]["Sample Database"], [-2.0], 0.5)
verifyStd(e["Solver"]["Sample Database"], [3.0], 0.5)
