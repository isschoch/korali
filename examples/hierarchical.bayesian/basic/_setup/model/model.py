#!/usr/bin/env python3
import sys
import os
import numpy as np


def normal_rnds(s, Ns):
  th1 = s["Parameters"][0]
  th2 = s["Parameters"][1]
  k = s["Sample Id"]
  y = np.random.normal(th1, th2, Ns)
  dataFolder = "_setup/data/"
  if not os.path.exists(dataFolder):
    os.makedirs(dataFolder)
  dataFile = dataFolder + "/data_set_" + str(k).zfill(3) + ".dat"
  np.savetxt(dataFile, np.transpose(y))


def getReferenceData(path, i):
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
