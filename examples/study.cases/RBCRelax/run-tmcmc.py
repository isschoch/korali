#!/usr/bin/env python3
import sys
import argparse

# Importing the computational model
import sys
sys.path.append('./model')
from model import *

# Creating new experiment
import korali
e = korali.Experiment()

procId = 0
procCount = 1
jobId = 0

if ('SLURM_PROCID' in os.environ): procId = os.environ['SLURM_PROCID']
if ('SLURM_NTASKS' in os.environ): procCount = os.environ['SLURM_NTASKS']
if ('SLURM_JOBID'  in os.environ): jobId = os.environ['SLURM_JOBID']

parser = argparse.ArgumentParser(prog='RBC Relaxation Sampling', description='Samples the viscosity parameter of an RBC inferred from actual experiments.')
parser.add_argument('--pop', help='Population Size. How many samples per generation.',  default=8)
parser.add_argument('--exp', help='Experiment name. Reference data will be taken from the {name}.txt file in the data folder.',  default='henon')
parser.add_argument('--lower', help='Lower bound for gammaC uniform prior', default=8000)
parser.add_argument('--upper', help='Upper bound for gammaC uniform prior', default=32000)
parser.add_argument('--tend', help='Value for tend parameter', default=0.4)
parser.add_argument('--inimesh_fname', help='Mesh file at init', default='stretch_Hen1999_d01.off')
args = parser.parse_args()

resFolder = "./results/tmcmc_" + args.exp 
profFile  = resFolder + '/profiling.' + str(jobId) + '.json'
refData   = getReferenceData(args.exp)
refPoints = getReferencePoints(args.exp)

ini_mesh_fname = "./data/off_files/{0}".format(args.inimesh_fname)

lowerBound = float(args.lower)
upperBound = float(args.upper)
expTend = float(args.tend)
popSize = float(args.pop)
expName = args.exp

if (int(procId) == int(procCount) - 1):
 print("[Korali] --------------------------------------------")
 print("[Korali] Running experiment: " + expName + ".txt ...")
 print("[Korali] Lower Gamma C Prior Bound: " + str(lowerBound))
 print("[Korali] Upper Gamma C Prior Bound: " + str(upperBound))
 print("[Korali] Tend Parameter: " + str(expTend))
 print("[Korali] Result Folder: " + resFolder)
 print("[Korali] Population Size: " + str(popSize))
 print("[Korali] Profiling File: " + profFile)
 print("[Korali] Job Id: " + str(jobId)) 
 print("[Korali] Rank Count: " + str(procCount))
 sys.stdout.flush()

# Setting up the reference likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"]   = refData

# Configuring TMCMC parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = popSize
e["Solver"]["Termination Criteria"]["Max Generations"] = 7 

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = lowerBound
e["Distributions"][0]["Maximum"] = upperBound

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

# Continue if results exists
# Loading previous results, if they exist.
found = e.loadState(resFolder + '/latest')
if (found == True):
 curMax = int(e["Solver"]["Termination Criteria"]["Max Generations"])
 newGen = curMax + 1
 if (int(procId) == int(procCount) - 1):
  print('[Korali] Found previous run at generation: ' + str(curMax))
  print('[Korali] Running now to generation: ' + str(newGen))
  sys.stdout.flush()
 e["Solver"]["Termination Criteria"]["Max Generations"] = newGen
 
# Setting Model after loading previous results to prevent bad function pointer
e["Problem"]["Computational Model"] = lambda sample: relaxModel(sample, refPoints, expName, expTend, ini_mesh_fname, korali.getMPIComm() )

# Configuring Linked (Distributed) Conduit
k = korali.Engine()
k["Conduit"]["Type"] = "Distributed"
k["Conduit"]["Workers Per Team"] = 2
k["Profiling"]["Detail"] = "Full"
k["Profiling"]["Path"] = profFile
k["Profiling"]["Frequency"] = 60

k.run(e)

