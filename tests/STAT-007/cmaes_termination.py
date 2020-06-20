#!/usr/bin/env python3
import os
import sys
import json
import korali
import argparse

sys.path.append('./helpers')
from helpers import *

#################################################
# CMAES run method
#################################################


def run_cmaes_with_termination_criterion(criterion, value):

  print("[Korali] Prepare CMAES run with Termination Criteria "\
          "'{0}'".format(criterion))

  e = korali.Experiment()

  e["Problem"]["Type"] = "Optimization"
  e["Problem"]["Objective Function"] = evaluateModel

  e["Variables"][0]["Name"] = "X"
  e["Variables"][0]["Lower Bound"] = +1.0
  e["Variables"][0]["Upper Bound"] = +10.0

  e["Solver"]["Type"] = "Optimizer/CMAES"
  e["Solver"]["Population Size"] = 8
  e["Solver"]["Termination Criteria"][criterion] = value

  e["Random Seed"] = 1337

  k = korali.Engine()
  k.run(e)

  if (criterion == "Max Generations"):
    assert_value(e["Current Generation"], value)

  elif (criterion == "Max Infeasible Resamplings"):
    assert_greatereq(e["Solver"]["Infeasible Sample Count"], value)

  elif (criterion == "Max Condition Covariance Matrix"):
    minEw = e["Solver"]["Minimum Covariance Eigenvalue"]
    maxEw = e["Solver"]["Maximum Covariance Eigenvalue"]
    assert_greatereq(maxEw / minEw, value)

  elif (criterion == "Max Value"):
    assert_greatereq(e["Solver"]["Best Ever Value"], value)

  elif (criterion == "Min Value Difference Threshold"):
    previous = e["Solver"]["Previous Best Ever Value"]
    current = e["Solver"]["Best Ever Value"]
    assert_smallereq(previous - current, value)

  elif (criterion == "Min Standard Deviation"):
    assert_smallereq(e["Solver"]["Current Min Standard Deviation"], value)

  elif (criterion == "Max Standard Deviation"):
    assert_greatereq(e["Solver"]["Current Max Standard Deviation"], value)

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

  run_cmaes_with_termination_criterion(args.criterion, args.value)
