#!/usr/bin/env python3

import numpy as np

N1 = 20
N2 = 20
N3 = 1000

f = lambda x: np.sin(x) * np.cos(3.0 * x) + np.cos(0.5 * x)

xTrain = np.random.uniform(0, 4 * np.pi, N1)
xTest = np.random.uniform(0, 4 * np.pi, N2)
xNew = np.random.uniform(0, 4 * np.pi, N3)

np.savetxt('sincos1d_train.dat', np.transpose([xTrain, f(xTrain)]))
np.savetxt('sincos1d_test.dat', np.transpose([xTest, f(xTest)]))
np.savetxt('sincos1d_new.dat', np.transpose([xNew, f(xNew)]))
