#!/usr/bin/env python3
import sys
import argparse

# Importing the computational model
import sys
sys.path.append('./model')
from model import *

# Creating new experiment
import korali

procId = 0
procCount = 1
jobId = 0

if ('SLURM_PROCID' in os.environ): procId = os.environ['SLURM_PROCID']
if ('SLURM_NTASKS' in os.environ): procCount = os.environ['SLURM_NTASKS']
if ('SLURM_JOBID'  in os.environ): jobId = os.environ['SLURM_JOBID']

parser = argparse.ArgumentParser(prog='RBC Relaxation Sampling', description='Samples the viscosity parameter of an RBC inferred from actual experiments.')
parser.add_argument('--pop', help='Population Size. How many samples per generation.',  default=512)
args = parser.parse_args()
popSize = int(args.pop)

expNames = [ 'hochmuth01', 'hochmuth02', 'hochmuth03', 'hochmuth04', 'henon' ]
eList = []

for i in range(len(expNames)):
 resFolder = "./results/tmcmc_multiple/" + expNames[i]
 profFile  = './results/tmcmc_multiple/profiling.' + str(jobId) + '.json'

 if (int(procId) == int(procCount) - 1):
  print("[Korali] --------------------------------------------")
  print("[Korali] Running experiment: " + expNames[i] + ".txt ...")
  print("[Korali] Result Folder: " + resFolder)
  print("[Korali] Profiling File: " + profFile)
  print("[Korali] TMCMC Population Size: " + str(popSize))
  print("[Korali] Job Id: " + str(jobId)) 
  print("[Korali] Rank Count: " + str(procCount))
  print("[Korali] Experiment Index: " + str(i))
 sys.stdout.flush()

 # Creating Experiment
 e = korali.Experiment()
 
 # Setting up the reference likelihood for the Bayesian Problem
 e["Problem"]["Type"] = "Bayesian/Reference" 
 e["Problem"]["Likelihood Model"] = "Normal"
 e["Problem"]["Reference Data"]   = getReferenceData(expNames[i])

 # Configuring TMCMC parameters
 e["Solver"]["Type"] = "Sampler/TMCMC"
 e["Solver"]["Population Size"] = popSize
 e["Solver"]["Termination Criteria"]["Max Generations"] = 10

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

 # Loading previous results, if they exist.
 e.loadState(resFolder + '/latest')
 eList.append(e)
 
eList[0]["Problem"]["Computational Model"] = lambda sample: relaxModel(sample, getReferencePoints('hochmuth01'), 'hochmuth01', 0.4, './data/off_files/stretch_H1979_d01.off'      , korali.getMPIComm() )
eList[1]["Problem"]["Computational Model"] = lambda sample: relaxModel(sample, getReferencePoints('hochmuth02'), 'hochmuth02', 0.4, './data/off_files/stretch_H1979_d02.off'      , korali.getMPIComm() )
eList[2]["Problem"]["Computational Model"] = lambda sample: relaxModel(sample, getReferencePoints('hochmuth03'), 'hochmuth03', 0.4, './data/off_files/stretch_H1979_d03.off'      , korali.getMPIComm() )
eList[3]["Problem"]["Computational Model"] = lambda sample: relaxModel(sample, getReferencePoints('hochmuth04'), 'hochmuth04', 0.4, './data/off_files/stretch_H1979_d04.off'      , korali.getMPIComm() )
eList[4]["Problem"]["Computational Model"] = lambda sample: relaxModel(sample, getReferencePoints('henon')     , 'henon'     , 0.4, './data/off_files/stretch_Hen1999_d01.off', korali.getMPIComm() )

# Configuring Linked (Distributed) Conduit
k = korali.Engine()
k["Conduit"]["Type"] = "Distributed"
k["Conduit"]["Workers Per Team"] = 2
k["Profiling"]["Detail"] = "Full"
k["Profiling"]["Path"] = profFile
k["Profiling"]["Frequency"] = 60

k.run(eList)
