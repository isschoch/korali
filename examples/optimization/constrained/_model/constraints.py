#!/usr/bin/env python

# Constraints


def g1(k):
  v = k["Parameters"]
  k["F(x)"] = -127.0 + 2 * v[0] * v[0] + 3.0 * pow(
      v[1], 4) + v[2] + 4.0 * v[3] * v[3] + 5.0 * v[4]


def g2(k):
  v = k["Parameters"]
  k["F(x)"] = -282.0 + 7.0 * v[0] + 3.0 * v[1] + 10.0 * v[2] * v[2] + v[3] - v[4]


def g3(k):
  v = k["Parameters"]
  k["F(x)"] = -196.0 + 23.0 * v[0] + v[1] * v[1] + 6.0 * v[5] * v[5] - 8.0 * v[6]


def g4(k):
  v = k["Parameters"]
  k["F(x)"] = 4.0 * v[0] * v[0] + v[1] * v[1] - 3.0 * v[0] * v[1] + 2.0 * v[
      2] * v[2] + 5.0 * v[5] - 11.0 * v[6]
