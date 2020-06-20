#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali
import argparse


parser = argparse.ArgumentParser(prog='RBC Relaxation w. Surrogate', description='Samples the viscosity parameter of an RBC inferred from actual experiments.')
parser.add_argument('--beta2', help='Covariance Scaling',  required = True)
parser.add_argument('--burnin', help='Default Burn In',  required = True)
parser.add_argument('--ub1', help='Upper Bound Psi3',  required = True)
parser.add_argument('--ub2', help='Upper Bound Psi 5',  required = True)

args = parser.parse_args()

ub1     = float(args.ub1)
ub2     = float(args.ub2)
beta2  = float(args.beta2)
burnin = int(args.burnin)

# Creating hierarchical Bayesian problem from previous two problems
e = korali.Experiment()

e["Problem"]["Type"]  = "Evaluation/Bayesian/Hierarchical/Psi"

# Hochmuth 1 - 5
for i in range(5):
  e["Problem"]["Subproblems"][i]["Path"]   = '../results_hierarchical/hochmuth' + str(i+1).zfill(2)
  e["Problem"]["Subproblems"][i]["Conditional Priors"] = ["Conditional 0", "Conditional 1"]

# Henon
e["Problem"]["Subproblems"][4]["Path"]   = '../results_hierarchical/henon'
e["Problem"]["Subproblems"][4]["Conditional Priors"] = ["Conditional 0", "Conditional 2"]

# Add probability of theta given psi, one per subproblem variable.
e["Variables"][0]["Name"] = "Psi 1"
e["Variables"][1]["Name"] = "Psi 2"
e["Variables"][2]["Name"] = "Psi 3"
e["Variables"][3]["Name"] = "Psi 4"
e["Variables"][4]["Name"] = "Psi 5"
e["Variables"][5]["Name"] = "Psi 6"

e["Variables"][0]["Prior Distribution"] = "Uniform 1"
e["Variables"][1]["Prior Distribution"] = "Uniform 2"
e["Variables"][2]["Prior Distribution"] = "Uniform 3"
e["Variables"][3]["Prior Distribution"] = "Uniform 4"
e["Variables"][4]["Prior Distribution"] = "Uniform 5"
e["Variables"][5]["Prior Distribution"] = "Uniform 6"

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Conditional 0"
e["Distributions"][0]["Type"] = "Univariate/Gaussian"
e["Distributions"][0]["Mean"] = "Psi 1"
e["Distributions"][0]["Standard Deviation"] = "Psi 2"

"""
e["Distributions"][1]["Name"] = "Conditional 1"
e["Distributions"][1]["Type"] = "Univariate/LogNormal"
e["Distributions"][1]["Mu"] = "Psi 3"
e["Distributions"][1]["Sigma"] = "Psi 4"

e["Distributions"][2]["Name"] = "Conditional 2"
e["Distributions"][2]["Type"] = "Univariate/LogNormal"
e["Distributions"][2]["Mu"] = "Psi 5"
e["Distributions"][2]["Sigma"] = "Psi 6"
"""

e["Distributions"][1]["Name"] = "Conditional 1"
e["Distributions"][1]["Type"] = "Univariate/Gamma"
e["Distributions"][1]["Shape"] = "Psi 3"
e["Distributions"][1]["Scale"] = "Psi 4"

e["Distributions"][2]["Name"] = "Conditional 2"
e["Distributions"][2]["Type"] = "Univariate/Gamma"
e["Distributions"][2]["Shape"] = "Psi 5"
e["Distributions"][2]["Scale"] = "Psi 6"


e["Distributions"][3]["Name"] = "Uniform 1"
e["Distributions"][3]["Type"] = "Univariate/Uniform"
e["Distributions"][3]["Minimum"] = 8000
e["Distributions"][3]["Maximum"] = 32000

e["Distributions"][4]["Name"] = "Uniform 2"
e["Distributions"][4]["Type"] = "Univariate/Uniform"
e["Distributions"][4]["Minimum"] = 0
e["Distributions"][4]["Maximum"] = 16000

e["Distributions"][5]["Name"] = "Uniform 3"
e["Distributions"][5]["Type"] = "Univariate/Uniform"
e["Distributions"][5]["Minimum"] = 0.0
e["Distributions"][5]["Maximum"] = ub1

e["Distributions"][6]["Name"] = "Uniform 4"
e["Distributions"][6]["Type"] = "Univariate/Uniform"
e["Distributions"][6]["Minimum"] = 0.0
e["Distributions"][6]["Maximum"] = 0.5

e["Distributions"][7]["Name"] = "Uniform 5"
e["Distributions"][7]["Type"] = "Univariate/Uniform"
e["Distributions"][7]["Minimum"] = 0.0
e["Distributions"][7]["Maximum"] = ub2

e["Distributions"][8]["Name"] = "Uniform 6"
e["Distributions"][8]["Type"] = "Univariate/Uniform"
e["Distributions"][8]["Minimum"] = 0.0
e["Distributions"][8]["Maximum"] = 1.0

e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 10000
e["Solver"]["Target Coefficient Of Variation"] = 0.6
e["Solver"]["Default Burn In"] = burnin
e["Solver"]["Covariance Scaling"] = beta2

e["Console"]["Verbosity"] = "Detailed"
e["Results"]["Frequency"] = 20
e["Results"]["Path"] = "results_surrogate_phase_2_{0}_{1}_{2}_{3}".format(ub1, ub2, beta2, burnin)

# Starting Korali's Engine and running experiment
k = korali.Engine()
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = 4
k.run(e)
