#!/usr/bin/env python3
import json


def evaluateLogLikelihood(s):
  v = s["Parameters"][0]
  r = -v * v
  s["logLikelihood"] = r


def assert_value(val, expected):
  assert val == expected, "Error: Value should be {0} but is {1}\n".format(
      expected, val)
