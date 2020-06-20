#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali

# Creating hierarchical Bayesian problem from previous two problems
e = korali.Experiment()

e["Problem"]["Type"]  = "Evaluation/Bayesian/Hierarchical/Theta"
e["Problem"]["Theta Problem Path"] = '../results_daint/tmcmc_hochmuth01'
e["Problem"]["Psi Problem Path"] = 'results_phase_2'

e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 1000

e["Console"]["Verbosity"] = "Detailed"
e["Results"]["Path"] = "results_phase_3b/"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = 4
k.run(e)
