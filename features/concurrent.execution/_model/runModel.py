#!/usr/bin/env python3
import sys
import subprocess


def runModel(x):
  argString = ['_model/model.py']
  v = x["Parameters"]
  for i in v:
    argString.append(str(i))
  retValue = subprocess.check_output(argString)
  result = float(retValue.decode())
  x["F(x)"] = result
