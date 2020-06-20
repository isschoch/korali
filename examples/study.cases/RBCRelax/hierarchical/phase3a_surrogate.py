#!/usr/bin/env python3

# Importing computational model
import korali

# Creating hierarchical Bayesian problem from previous two problems
e = korali.Experiment()

e["Problem"]["Type"]  = "Evaluation/Bayesian/Hierarchical/ThetaNew"
#e["Problem"]["Psi Problem Path"] = 'results_surrogate_phase_2'
e["Problem"]["Psi Problem Path"] = 'results_surrogate_phase_2_10.0_5.0_0.01_3/'
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 8000
e["Distributions"][0]["Maximum"] = 32000

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.0
e["Distributions"][1]["Maximum"] = 1.0

e["Distributions"][2]["Name"] = "Uniform 2"
e["Distributions"][2]["Type"] = "Univariate/Uniform"
e["Distributions"][2]["Minimum"] = 0.0
e["Distributions"][2]["Maximum"] = 1.0

e["Variables"][0]["Name"] = "gammaC"
e["Variables"][1]["Name"] = "sigma1"
e["Variables"][2]["Name"] = "sigma2"

e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"
e["Variables"][2]["Prior Distribution"] = "Uniform 2"

e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 10000
#e["Solver"]["Target Coefficient Of Variation"] = 0.6
e["Solver"]["Default Burn In"] = 9
e["Solver"]["Covariance Scaling"] = 0.02


e["Console"]["Verbosity"] = "Detailed"
e["Results"]["Path"] = "results_surrogate_phase_3a_paper9_0.02/"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = 4
k.run(e)
