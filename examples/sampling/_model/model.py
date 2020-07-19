#!/usr/bin/env python


# This is the negative square -(x^2)
def model(s):
  v = s["Parameters"][0]
  s["logP(x)"] = -0.5 * v * v
  s["grad(logP(x))"] = [-v]
