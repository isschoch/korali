#!/usr/bin/env python3
import korali
import sys

sys.path.append("setup/")
from runLAMMPS import *

k = korali.Engine()
e = korali.Experiment()

# Defining Concurrent Jobs
cJobs = 1
if (len(sys.argv) > 1): cJobs = int(sys.argv[1])

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = getReferenceData()
e["Problem"]["Computational Model"] = runLAMMPS

e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 2.8
e["Distributions"][0]["Maximum"] = 3.0

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.5
e["Distributions"][1]["Maximum"] = 1.25

e["Distributions"][2]["Name"] = "Uniform 2"
e["Distributions"][2]["Type"] = "Univariate/Uniform"
e["Distributions"][2]["Minimum"] = 0.0
e["Distributions"][2]["Maximum"] = 0.1

# Configuring the problem's variables and their prior distributions
e["Variables"][0]["Name"] = "Model-Epsilon"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["Variables"][1]["Name"] = "Model-Sigma"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"

e["Variables"][2]["Name"] = "[Sigma]"
e["Variables"][2]["Prior Distribution"] = "Uniform 2"

# Configuring TMCMC parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Max Generations"] = 20

# General Settings
e["File Output"]["Path"] = "_result_lammps_tmcmc_" + str(cJobs).zfill(4) + "_workers"
e["Console Output"]["Verbosity"] = "Detailed"

# Selecting external conduit
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = cJobs

k.run(e)
