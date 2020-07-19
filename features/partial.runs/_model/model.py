#!/usr/bin/env python
import os
import sys
import numpy as np


# This is the negative square -(x^2)
def model(s):
  x = s["Parameters"][0]
  r = -0.5 * x * x
  s["F(x)"] = r
  s["logP(x)"] = r


# Function and Gradient function evaluation
def model_with_gradient(p):
  X = p["Parameters"]
  x = X[0]
  gradient = []
  evaluation = -0.5 * x * x
  gradient.append(-x)
  p["F(x)"] = evaluation
  p["Gradient"] = gradient


def calculateLogLikelihood(s):
  x = s["Parameters"][0]
  r = -0.5 * x * x
  s["logLikelihood"] = r


def prepareFile(fileName):
  if os.path.isdir('_propagation_output') == False:
    os.mkdir('_propagation_output')
    os.system('touch {0}'.format(fileName))


def put_normal_rnds(theta, Ns, fileName):
  mu = theta["Parameters"][0]
  var = theta["Parameters"][1]

  y = np.random.normal(mu, var, Ns)

  if os.path.isfile(fileName):
    f = open(fileName, 'a+')
    np.savetxt(f, np.transpose(y))
    f.close()

  else:
    sys.exit(
        'put_normal_rnds: file \'{0}\' does not exist! exit..'.format(fileName))
