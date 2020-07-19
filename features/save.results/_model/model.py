#!/usr/bin/env python3
import math

# Allocating storage for Quantities of Interest
# In this case we use a dictionary that associates the
# quantities of interests to a specific sample Id, provided
# by Korali.
samplePHs = {}
sampleTemps = {}
bananaCounts = {}
orangeCounts = {}
appleCounts = {}
harvestQualities = {}

# We run the computational model for each sample, storing
# individual fruit counts, but also returning the harvest
# quality back to Korali to sample around the parameter
# space with higher yield.

# Our computational model calculates the harvest yield of
# bananas, apples, and oranges from a farming field with
# respect to the soil pH and average temperatures.

# The main quantity of interest is the 'harvest quality',
# which measures the nutritional value of the harvest as
# a whole. The more produce, the better harvest. This is
# the value we want to sample from.

# However, we also want to store the number of individual
# fruit counts during our sampling. In this example, we
# see how to keep these values.


def storeResults(d):
  # Gathering sample variable values
  ph = d["Parameters"][0]
  temp = d["Parameters"][1]
  sampleId = d["Sample Id"]

  # Gathering QOIs
  a = getAppleCount(ph, temp)
  b = getBananaCount(ph, temp)
  o = getOrangeCount(ph, temp)

  # Calculating objective function
  x = getHarvestQuality(b, o, a)

  # Storing QOIS
  samplePHs[sampleId] = ph
  sampleTemps[sampleId] = temp
  bananaCounts[sampleId] = b
  orangeCounts[sampleId] = o
  appleCounts[sampleId] = a
  harvestQualities[sampleId] = x

  # Store QoI
  d["Apples"] = a
  d["Bananas"] = b

  # Returning evaluation to Korali
  d["logP(x)"] = x


def getHarvestQuality(b, o, a):
  return b * 1.2 + o * 0.9 + a * 1.1


def getBananaCount(ph, temp):
  a = 100
  b = ((ph - 7.3))**2 + ((temp - 35) * 0.1)**2
  c = 2
  t = a * math.exp(-b / c)
  return math.floor(t)


def getOrangeCount(ph, temp):
  a = 50
  b = ((ph - 6.6))**2 + ((temp - 27) * 0.1)**2
  c = 2
  t = a * math.exp(-b / c)
  return math.floor(t)


def getAppleCount(ph, temp):
  a = 90
  b = ((ph - 8.6))**2 + ((temp - 20) * 0.1)**2
  c = 2
  t = a * math.exp(-b / c)
  return math.floor(t)
