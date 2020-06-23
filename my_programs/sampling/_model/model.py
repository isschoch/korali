#!/usr/bin/env python


# This is the negative square -(x^2)
def model(s):
  v = s["Parameters"][0]
  s["P(x)"] = -0.5 * v * v
