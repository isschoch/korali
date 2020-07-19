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
e["Console Output"]["Frequency"] = 500
e["File Output"]["Frequency"] = 500

e["Problem"]["Type"] = "Sampling"
e["Problem"]["Probability Function"] = model

# Configuring the HMC sampler parameters
e["Solver"]["Type"] = "Sampler/HMC"
e["Solver"]["Burn In"] = 500
e["Solver"]["Termination Criteria"]["Max Samples"] = 5000

# HMC specific parameters
e["Solver"]["Use Adaptive Step Size"] = 1
e["Solver"]["Use Euclidean Metric"] = 1
e["Solver"]["Use NUTS"] = 1
e["Solver"]["Num Integration Steps"] = 20
e["Solver"]["Step Size"] = 0.1
e["Solver"]["Target Integration Time"] = 1.0
e["Solver"]["Desired Average Acceptance Rate"] = 0.8

# Defining problem's variables and their HMC settings
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Mean"] = 1.0
e["Variables"][0]["Initial Standard Deviation"] = 1.0

# Running Korali
e["Random Seed"] = 0xC0FFEE
e["File Output"]["Path"] = "_result_run-hmc-nuts"

k = korali.Engine()
k.run(e)

# Testing Results
checkMean(e, 0.0, 0.01)
checkStd(e, 1.0, 0.025)
