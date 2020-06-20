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

e["Random Seed"] = 0xC0FFEE
e["File Output"]["Path"] = "_result_run-tmcmc"

# Configuring problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = model

e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -20.0
e["Distributions"][0]["Maximum"] = +20.0

# Defining problem's variables and prior distribution for TMCMC
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

# Configuring the TMCMC sampler parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 5000
e["Solver"]["Target Coefficient Of Variation"] = 0.6

# Running Korali
k.run(e)

checkMean(e, 0.0, 0.05)
checkStd(e, 1.0, 0.025)
