#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

scaling = 5.0

import korali
k = korali.Engine()

# Defining Training Sets
np.random.seed(0xC0FFEE)

trainingInputSet = np.random.uniform(0, 2 * np.pi, 500)
trainingSolutionSet = np.sin(trainingInputSet) * scaling

trainingInputSet = [ [ i ] for i in trainingInputSet.tolist() ]
trainingSolutionSet = [ [ i ] for i in trainingSolutionSet.tolist() ]

e = korali.Experiment()

### Defining a learning problem to infer values of sin(x)

e["Problem"]["Type"] = "Supervised Learning"

e["Problem"]["Inputs"] = trainingInputSet
e["Problem"]["Outputs"] = trainingSolutionSet

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "Learner/DeepGD"
e["Solver"]["Steps Per Generation"] = 100
e["Solver"]["Batch Normalization"]["Enabled"] = True
e["Solver"]["Optimizer"]["Type"] = "Optimizer/Adam"

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Layers"][0]["Type"] = "Input"
e["Solver"]["Neural Network"]["Layers"][0]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Identity"

e["Solver"]["Neural Network"]["Layers"][1]["Type"] = "Dense"
e["Solver"]["Neural Network"]["Layers"][1]["Node Count"] = 5
e["Solver"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Tanh"

e["Solver"]["Neural Network"]["Layers"][2]["Type"] = "Output"
e["Solver"]["Neural Network"]["Layers"][2]["Node Count"] = 1
e["Solver"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Identity"

e["Console Output"]["Frequency"] = 1
e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Training the neural network

e["Solver"]["Termination Criteria"]["Max Generations"] = 100

k.resume(e)

### Obtaining inferred results from the NN and comparing them to the actual solution

testInputSet = np.random.uniform(0, 2 * np.pi, 100)
testInputSet = [[x] for x in testInputSet.tolist()]

testInferredSet = e.getEvaluation(testInputSet)
testGradientSet = e.getGradients(testInferredSet)
testOutputSet = np.sin(testInputSet) * scaling

### Plotting Results

plt.plot(testInputSet, testOutputSet, "o")
plt.plot(testInputSet, testInferredSet, "x")
#plt.plot(testInferredSet, testGradientSet, "*")
plt.show()
