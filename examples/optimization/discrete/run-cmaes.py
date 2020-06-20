#!/usr/bin/env python3

# In this example, we demonstrate how Korali finds values for the
# variables that maximize the objective function, given by a
# user-provided computational model, subject to a set of
# constraints.

# Importing the computational model and constraints
import sys
sys.path.append('_model')
from model import *

# Creating new experiment
import korali
e = korali.Experiment()

# Selecting problem type
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = model

# Creating 10 variables and setting their CMA-ES bounds
for i in range(10):
  e["Variables"][i]["Name"] = "X" + str(i)
  e["Variables"][i]["Initial Value"] = 1.0
  e["Variables"][i]["Lower Bound"] = -19.0
  e["Variables"][i]["Upper Bound"] = +21.0

# We set some of them as discrete.
e["Variables"][0]["Granularity"] = 1.0
e["Variables"][1]["Granularity"] = 1.0
e["Variables"][3]["Granularity"] = 1.0
e["Variables"][6]["Granularity"] = 1.0

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-9
e["Solver"]["Termination Criteria"]["Max Generations"] = 5000

# Configuring output settings
e["File Output"]["Frequency"] = 50
e["Console Output"]["Frequency"] = 50

# Starting Korali's Engine and running experiment
k = korali.Engine()
k.run(e)
