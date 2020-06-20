#!/usr/bin/env python3
import sys
sys.path.append("model")
import data
import model
import korali

k = korali.Engine()
e = korali.Experiment()

# Defining Concurrent Jobs
popSize = 512
if (len(sys.argv) > 1): popSize = int(sys.argv[1])

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = data.getReferenceData().tolist()
e["Problem"]["Computational Model"] = lambda koraliData: model.evaluate( koraliData, data.getReferencePoints(), korali.getMPIComm() )

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

e["Variables"][1]["Name"] = "ks"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"

e["Variables"][2]["Name"] = "[Sigma]"
e["Variables"][2]["Prior Distribution"] = "Uniform 2"

# Setting CMAES as Solver
e["Solver"]["Type"]  = "Sampler/TMCMC"
e["Solver"]["Population Size"] = int(popSize)
e["Solver"]["Termination Criteria"]["Max Generations"] = 20

# General Settings
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Path"] = "_result_rbc_tmcmc_" + str(popSize).zfill(4) + "_nodes"
e["Store Sample Information"] = True

# Configuring Distributed Conduit
k["Conduit"]["Type"] = "Distributed"
k["Conduit"]["Workers Per Team"] = 1
k["Profiling"]["Detail"] = "Full"
k["Profiling"]["Path"] =  "_result_rbc_tmcmc_" + str(popSize).zfill(4) + "_nodes/profiling.json"
k["Profiling"]["Frequency"] = 60

k.run(e)
