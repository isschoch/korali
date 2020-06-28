#!/usr/bin/env python


# This is the negative square -(x^2)
def model(s):
  v1 = s["Parameters"][0]
  v2 = s["Parameters"][1]
  v3 = s["Parameters"][2]
  s["P(x)"] = -0.5 * (v1 * v1 + v2 * v2 + v3 * v3)
