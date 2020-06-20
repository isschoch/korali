#!/usr/bin/env python3
import sys
import os
import numpy as np


def logistic(X, s):
  th1 = s["Parameters"][0]
  th2 = s["Parameters"][1]
  th3 = s["Parameters"][2]
  sig = s["Parameters"][3]

  result = []
  sdev = []

  for x in X:
    f = np.exp(th3 * x)
    y = (th1 * th2 * f) / (th1 + th2 * (f - 1))
    result.append(y)
    sdev.append(sig)

  s["Reference Evaluations"] = result
  s["Standard Deviation"] = sdev


def logistic_reference(s):
  th = np.zeros(4)
  for i in range(4):
    th[i] = s["Parameters"][i]

  X = np.linspace(0.0, 10.0, num=21)
  Y = np.zeros(X.size)
  for i in range(X.size):
    f = np.exp(th[2] * X[i])
    if (th[0] * th[1] * f == 0):
      Y[i] = 0.0
    else:
      Y[i] = (th[0] * th[1] * f) / (th[0] + th[1] * (f - 1))

  Y = Y + np.random.normal(0, th[3], X.size)
  k = s["Sample Id"]

  dataFolder = "_setup/data/"
  if not os.path.exists(dataFolder):
    os.makedirs(dataFolder)
  dataFile = dataFolder + "/data_set_" + str(k).zfill(3) + ".dat"
  np.savetxt(dataFile, np.transpose([X, Y]))


def getReferenceData(path, i):
  fileName = path + "/data_set_" + str(i).zfill(3) + ".dat"
  y = readColumnFromFile(fileName, 1)
  return y


def getReferencePoints(path, i):
  fileName = path + "/data_set_" + str(i).zfill(3) + ".dat"
  y = readColumnFromFile(fileName, 0)
  return y


def readColumnFromFile(FileName, Column):
  try:
    f = open(FileName, "r")
    lines = f.readlines()
    y = []
    for l in lines:
      y.append(float(l.split()[Column]))
    f.close()
  except IOError as e:
    print("I/O error(" + str(e.errno) + "): " + e.strerror)
    print("The file " + FileName + " is missing")
    sys.exit(1)
  except ValueError:
    print("Could not convert data to a float. Check the results in " + FileName)
    sys.exit(1)
  except:
    print("Unexpected error: " + sys.exc_info()[0])
    raise

  return y
