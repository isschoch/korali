#!/usr/bin/env python3

# In this example, we demonstrate how Korali finds values for the
# variables that maximize the objective function, given by a
# user-provided computational model.

# Importing computational model
import sys
sys.path.append('./_model')
from model import *

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()

# Configuring Problem
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = model

# Configuring Solver
e["Solver"]["Type"] = "Optimizer/DEA"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-12
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Defining the problem's variables and their bounds.
e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

# Configuring results path
e["File Output"]["Path"] = '_korali_result_dea'

# Running Korali
k.run(e)
