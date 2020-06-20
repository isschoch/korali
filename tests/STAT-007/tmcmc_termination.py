#!/usr/bin/env python3
import os
import sys
import json
import korali
import argparse

sys.path.append('./helpers')
from helpers import *

#################################################
# TMCMC run method
#################################################


def run_tmcmc_with_termination_criterion(criterion, value):

  print("[Korali] Prepare TMCMC run with Termination Criteria "\
          "'{0}'".format(criterion))

  e = korali.Experiment()
  e["Problem"]["Type"] = "Bayesian/Custom"
  e["Problem"]["Likelihood Model"] = evaluateLogLikelihood

  e["Distributions"][0]["Name"] = "Uniform 0"
  e["Distributions"][0]["Type"] = "Univariate/Uniform"
  e["Distributions"][0]["Minimum"] = -10.0
  e["Distributions"][0]["Maximum"] = +10.0

  e["Variables"][0]["Name"] = "X"
  e["Variables"][0]["Prior Distribution"] = "Uniform 0"

  e["Solver"]["Type"] = "Sampler/TMCMC"
  e["Solver"]["Population Size"] = 5000
  e["Solver"]["Covariance Scaling"] = 0.001
  e["Solver"]["Termination Criteria"][criterion] = value

  e["Random Seed"] = 1337

  k = korali.Engine()
  k.run(e)

  if (criterion == "Max Generations"):
    assert_value(e["Current Generation"], value)

  elif (criterion == "Target Annealing Exponent"):
    assert_greatereq(e["Solver"]["Annealing Exponent"], value)

  else:
    print("Termination Criterion not recognized!")
    exit(-1)


#################################################
# Main (called from run_test.sh with args)
#################################################

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      prog='cmaes_termination', description='Check Termination Criterion.')
  parser.add_argument(
      '--criterion',
      help='Name of Termination Criterion',
      action='store',
      required=True)
  parser.add_argument(
      '--value',
      help='Value of Termination Criterion',
      action='store',
      type=float,
      required=True)
  args = parser.parse_args()

  run_tmcmc_with_termination_criterion(args.criterion, args.value)
