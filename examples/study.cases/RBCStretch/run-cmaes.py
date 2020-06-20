#!/usr/bin/env python3
import sys
sys.path.append("model")
import data
import model
import korali

k = korali.Engine()
e = korali.Experiment()

# Defining Concurrent Jobs
cJobs = 1
if (len(sys.argv) > 1): cJobs = int(sys.argv[1]) 

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = data.getReferenceData().tolist()
e["Problem"]["Computational Model"] = lambda koraliData: model.evaluate(koraliData, data.getReferencePoints().tolist(), k.getMPIComm())

e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.2
e["Distributions"][0]["Maximum"] = 0.6

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 10.0
e["Distributions"][1]["Maximum"] = 40.0

e["Distributions"][2]["Name"] = "Uniform 2"
e["Distributions"][2]["Type"] = "Univariate/Uniform"
e["Distributions"][2]["Minimum"] = 1e-5
e["Distributions"][2]["Maximum"] = 2.0

# Setting up Problem Variables
e["Variables"][0]["Name"] = "x0"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Value"] = 0.4
e["Variables"][0]["Initial Standard Deviation"] = 0.05

e["Variables"][1]["Name"] = "ks"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"
e["Variables"][1]["Initial Value"] = 25.0
e["Variables"][1]["Initial Standard Deviation"] = 3.0

e["Variables"][2]["Name"] = "[Sigma]"
e["Variables"][2]["Prior Distribution"] = "Uniform 2"
e["Variables"][2]["Initial Value"] = 1.0
e["Variables"][2]["Initial Standard Deviation"] = 0.05

# Setting CMAES as Solver
e["Solver"]["Type"]  = "Optimizer/CMAES"
e["Solver"]["Population Size"] = int(cJobs)
e["Solver"]["Termination Criteria"]["Max Generations"] = 50

# General Settings
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Path"] = "_result_rbc_cmaes_" + str(cJobs).zfill(4) + "_nodes"

# Configuring Distributed Conduit
k["Conduit"]["Type"] = "Distributed"
k["Conduit"]["Workers Per Team"] = 1
k.run(e)
