#!/usr/bin/env python3

import os
import sys
import csv
import korali
import numpy as np

from os import listdir
from os.path import isfile, join

train_data = np.loadtxt('_data/sincos1d_train.dat', usecols=range(2))
test_data = np.loadtxt('_data/sincos1d_test.dat', usecols=range(2))

trainInput    = [ [ i ] for i in train_data[:, 0].tolist() ]
trainSolution = [ [ i ] for i in train_data[:, 1].tolist() ]

import korali
k = korali.Engine()
e = korali.Experiment()

e['Problem']['Type'] = 'Supervised Learning'

e["Problem"]["Inputs"] = trainInput
e["Problem"]["Outputs"] = trainSolution

e['Solver']['Type'] = 'Learner/Gaussian Process'
e['Solver']['Covariance Function'] = 'CovSum ( CovSEiso, CovNoise)'

e['Solver']['Optimizer']['Type'] = 'Optimizer/Rprop'
e['Solver']['Optimizer']['Termination Criteria']['Max Generations'] = 1000
e['Solver']['Optimizer']['Termination Criteria'][
    'Parameter Relative Tolerance'] = 1e-8

e['Console Output']['Verbosity'] = 'Normal'
e['Console Output']['Frequency'] = 10

e['File Output']['Enabled'] = False
e['Random Seed'] = 0xC0FFEE

k.run(e)

x = np.linspace(0, 14, 1000)
x = [[i] for i in x.tolist()]
y = e.getEvaluation(x)

z = [i + j for i, j in zip(x, y)]

with open('_data/results.dat', 'w') as f:
  wr = csv.writer(f, delimiter=' ')
  wr.writerows(z)
