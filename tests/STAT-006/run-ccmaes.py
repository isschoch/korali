#!/bin/bash
import os
import sys
import json
import korali
import argparse

sys.path.append('./helpers')
from helpers import *

#################################################
# CCMAES run method
#################################################


def run_ccmaes(constraint):

  print("[Korali] Prepare CMAES run with Termination Criteria "\
          "'{0}'".format(constraint))

  e = korali.Experiment()

  e["Problem"]["Type"] = "Optimization"
  e["Problem"]["Objective Function"] = evaluateModel

  e["Variables"][0]["Name"] = "X"
  e["Variables"][0]["Lower Bound"] = -10.0
  e["Variables"][0]["Upper Bound"] = +10.0

  e["Variables"][1]["Name"] = "Y"
  e["Variables"][1]["Lower Bound"] = -10.0
  e["Variables"][1]["Upper Bound"] = +10.0

  e["Solver"]["Type"] = "Optimizer/CMAES"
  e["Solver"]["Population Size"] = 8
  e["Solver"]["Viability Population Size"] = 2
  e["Solver"]["Termination Criteria"]["Max Generations"] = 100
  e["Solver"]["Is Sigma Bounded"] = 1

  e["Console Output"]["Verbosity"] = "Detailed"
  e["Random Seed"] = 1337

  k = korali.Engine()

  if (constraint == "None"):
    k.run(e)
    assert_greatereq(e["Solver"]["Best Ever Value"], -6 * 1e-10)

  elif (constraint == "Inactive"):
    e["Problem"]["Constraints"] = [inactive1, inactive2]
    k.run(e)
    assert_greatereq(e["Solver"]["Best Ever Value"], -1.8 * 1e-10)

  elif (constraint == "Active at Max 1"):
    e["Problem"]["Constraints"] = [activeMax1, activeMax2]
    k.run(e)
    assert_greatereq(e["Solver"]["Best Ever Value"], -4.826824e+00)

  elif (constraint == "Active at Max 2"):
    e["Problem"]["Constraints"] = [
        activeMax1, activeMax2, activeMax3, activeMax4
    ]
    k.run(e)
    assert_greatereq(e["Solver"]["Best Ever Value"], -9.653645e+00)

  elif (constraint == "Inactive at Max 1"):
    e["Problem"]["Constraints"] = [inactiveMax1, inactiveMax2]
    k.run(e)
    assert_greatereq(e["Solver"]["Best Ever Value"], -2.19963e-10)

  elif (constraint == "Inactive at Max 2"):
    e["Problem"]["Constraints"] = [
        inactiveMax1, inactiveMax2, inactiveMax3, inactiveMax4
    ]
    k.run(e)
    assert_greatereq(e["Solver"]["Best Ever Value"], -4.626392e-10)

  elif (constraint == "Mixed"):
    e["Problem"]["Constraints"] = [
        activeMax1, activeMax2, activeMax3, activeMax4, inactiveMax1,
        inactiveMax2, inactiveMax3, inactiveMax4
    ]
    k.run(e)
    assert_greatereq(e["Solver"]["Best Ever Value"], -7.895685e+01)

  elif (constraint == "Stress"):
    e["Problem"]["Constraints"] = [
        activeMax1, activeMax2, activeMax3, activeMax4, inactiveMax1,
        inactiveMax2, inactiveMax3, inactiveMax4, stress1, stress2, stress3,
        stress4, stress5, stress6, stress7, stress8
    ]
    k.run(e)
    assert_greatereq(e["Solver"]["Best Ever Value"], -7.895685e+01)

  else:
    print("Constraint not recognized!")
    exit(-1)


#################################################
# Main (called from run_test.sh with arg)
#################################################

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      prog='cmaes_termination', description='Check Termination Criterion.')
  parser.add_argument(
      '--constraint', help='Name of Constraint', action='store', required=True)
  args = parser.parse_args()

  run_ccmaes(args.constraint)
