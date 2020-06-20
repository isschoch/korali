#!/usr/bin/env python3
import sys
import argparse

# Importing the computational model
import sys
sys.path.append('./model')
from surrogate import *

# Creating new experiment
import korali

parser = argparse.ArgumentParser(prog='RBC Relaxation Sampling', description='Samples the viscosity parameter of an RBC inferred from actual experiments.')
parser.add_argument('--pop', help='Population Size. How many samples per generation.',  default=512)
args = parser.parse_args()
popSize = int(args.pop)

expNames = [ 'hochmuth01', 'hochmuth02', 'hochmuth03', 'hochmuth04', 'henon' ]

# cm=[]
# cm.append( lambda sample: relaxModel_surrogate(sample, getReferencePoints('hochmuth01'), 'hochmuth01', 0.4 ) )
# cm.append( lambda sample: relaxModel_surrogate(sample, getReferencePoints('hochmuth02'), 'hochmuth02', 0.4 ) )
# cm.append( lambda sample: relaxModel_surrogate(sample, getReferencePoints('hochmuth03'), 'hochmuth03', 0.4 ) )
# cm.append( lambda sample: relaxModel_surrogate(sample, getReferencePoints('hochmuth04'), 'hochmuth04', 0.4 ) )
# cm.append( lambda sample: relaxModel_surrogate(sample, getReferencePoints('henon'), 'henon', 0.4 ) )

# for i in range(len(expNames)):
#
i=0
resFolder = "./results/tmcmc_surrogate_multiple/" + expNames[i]

print("[Korali] --------------------------------------------")
print("[Korali] Running experiment: " + expNames[i] + ".txt ...")
print("[Korali] Result Folder: " + resFolder)
print("[Korali] TMCMC Population Size: " + str(popSize))
print("[Korali] Experiment Index: " + str(i))
sys.stdout.flush()



# Creating Experiment
e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal" 
e["Problem"]["Reference Data"]   = getReferenceData(expNames[i])

e["Problem"]["Computational Model"] = lambda sample: relaxModel_surrogate(sample, getReferencePoints('hochmuth01'), 'hochmuth01', 0.4 )

# Configuring TMCMC parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = popSize
e["Solver"]["Target Coefficient Of Variation"] = 1.0
e["Solver"]["Covariance Scaling"] = 0.04

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 8000
e["Distributions"][0]["Maximum"] = 32000

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.0
e["Distributions"][1]["Maximum"] = 1.0

# Configuring the problem's variables
e["Variables"][0]["Name"] = "gammaC"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["Variables"][1]["Name"] = "[Sigma]"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"

# General Settings
e["Console Output"]["Verbosity"]  = "Detailed"
e["File Output"]["Path"]          = resFolder
e["Store Sample Information"] = True

k = korali.Engine()
k.run(e)
