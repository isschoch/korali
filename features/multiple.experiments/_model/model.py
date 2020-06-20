#!/usr/bin/env python

# This is a linear regression model with two params (slope and intercept)


def model(s, x):
  v1 = s["Parameters"][0]
  v2 = s["Parameters"][1]
  sig = s["Parameters"][2]

  result = []
  sdev = []
  for i in range(len(x)):
    result.append(v1 * x[i] + v2)
    sdev.append(sig)

  s["Reference Evaluations"] = result
  s["Standard Deviation"] = sdev


def getReferenceData():
  y = []
  y.append(3.2069)
  y.append(4.1454)
  y.append(4.9393)
  y.append(6.0588)
  y.append(6.8425)
  return y


def getReferencePoints():
  x = []
  x.append(1.0)
  x.append(2.0)
  x.append(3.0)
  x.append(4.0)
  x.append(5.0)
  return x
