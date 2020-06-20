#! /usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from korali.plotter.helpers import hlsColors, drawMulticoloredLine
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def plotVariables(genList):
  numgens = len(genList)

  lastGen = 0
  generations = []
  for i in genList:
    generations.append(genList[i]["Current Generation"])
    if genList[i]['Current Generation'] > lastGen:
      lastGen = genList[i]['Current Generation']

  some_gen = list(genList.keys())[0]
  numLatent = genList[some_gen]["Solver"]["Number Latent"]
  numHyper = genList[some_gen]["Solver"]["Number Hyperparameters"]
  numSamples = genList[some_gen]["Solver"]["Number Samples Per Step"]
  sample_means = np.zeros((len(genList), numLatent))
  sample_sdevs = np.zeros((len(genList), numLatent))
  hyperparams = np.zeros((len(genList), numHyper))

  # determine which are latent, vs. hyperparameters
  latentVars = []
  hyperVars = []
  for var_dict in genList[some_gen]["Variables"]:
    if var_dict["Bayesian Type"] == "Hyperparameter":
      hyperVars.append(var_dict)
    else:
      latentVars.append(var_dict)

  for idx, i in enumerate(genList):
    sample_means[idx, ...] = genList[i]["Solver"]["Current Sample Means"]
    sample_sdevs[
        idx, ...] = genList[i]["Solver"]["Current Sample Standard Deviations"]
    hyperparams[idx, ...] = genList[i]["Solver"]["Current Hyperparameters"]

  fig, ax = plt.subplots(
      nrows=max(numLatent, numHyper) + 1, ncols=2, figsize=(8, 20), dpi=60)
  fig.subplots_adjust(hspace=.8)
  # samplesTmp = np.reshape( samples, (numentries,numdim) )
  plt.suptitle(
      'SAEM Plotter - \nNumber of Samples {0}\n'.format(str(numSamples)),
      fontweight='bold',
      fontsize=12)
  for latentIdx in range(numLatent):
    ax[latentIdx, 0].errorbar(
        x=generations,
        y=sample_means[:, latentIdx],
        yerr=sample_sdevs[:, latentIdx])
    ax[latentIdx, 0].set_title(latentVars[latentIdx]["Name"])
  # fig, ax = plt.subplots(nrows=max(numLatent, numHyper) + 1, ncols=1, figsize=(15,8))
  for hyperIdx in range(numHyper):
    ax[hyperIdx, 1].plot(generations, hyperparams[:, hyperIdx])
    ax[hyperIdx, 1].set_title(hyperVars[hyperIdx]["Name"])
  ax[-1, 0].set_title("gamma")

  return fig, ax


def plot(genList, args):

  fig, ax = plotVariables(genList)
