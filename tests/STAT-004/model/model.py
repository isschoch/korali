#!/usr/bin/env python
import math
import numpy as np

# log Gaussian with mean -2.0 and var 3.0
def lgaussian(s):
  x0 = s["Parameters"][0]
  r = -0.5 * ((x0 + 2.0)**2 / (9.0)) - 0.5 * math.log(2 * math.pi * 9)
  s["logP(x)"] = r
  s["grad(logP(x))"] = [-(x0 + 2.0) / 9.0]


def lgaussianCustom(s):
  x0 = s["Parameters"][0]
  r = -0.5 * ((x0 + 2.0)**2 / (9.0)) - 0.5 * math.log(2 * math.pi * 9)
  s["logLikelihood"] = r


# log Gaussian in d dimension with mean 0 and var 1
def lgaussianxd(s, d):
  ss = 0.0
  for i in range(d):
    ss += s["Parameters"][i]**2
  r = -0.5 * ss
  s["logP(x)"] = r
  grad = []
  for i in range(d):
    grad.append(-s["Parameters"][i])
  s["grad(logP(x))"] = grad


def lgaussianxdCustom(s, d):
  ss = 0.0
  for i in range(d):
    ss += s["Parameters"][i]**2
  r = -0.5 * ss
  s["logLikelihood"] = r


# log exponential with mean 4
def lexponential(s):
  lam = 1. / 4.
  x0 = s["Parameters"][0]
  r = 0.0
  if (x0 < 0):
    r = -math.inf
  else:
    r = math.log(lam) - lam * x0
  s["logP(x)"] = r
  s["grad(logP(x))"] = [-lam]


def lexponentialCustom(s):
  lam = 1. / 4.
  x0 = s["Parameters"][0]
  r = 0.0
  if (x0 < 0):
    r = -math.inf
  else:
    r = math.log(lam) - lam * x0
  s["logLikelihood"] = r


# log laplace with mean 4 and scale 1 (var 2)
def llaplace(s):
  x0 = s["Parameters"][0]
  mu = 4.0
  scale = 1
  r = -math.log(2.0 * scale) - abs(x0 - mu) / scale
  s["logP(x)"] = r
  s["grad(logP(x))"] = [-np.sign(x0 - mu) / scale]


def llaplaceCustom(s):
  x0 = s["Parameters"][0]
  mu = 4.0
  scale = 1
  r = -math.log(2.0 * scale) - abs(x0 - mu) / scale
  s["logLikelihood"] = r
