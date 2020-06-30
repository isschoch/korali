#!/usr/bin/env python
import numpy as np

# This is the negative square -(x^2)
def model(s):
  # v1 = s["Parameters"][0]
  # v2 = s["Parameters"][1]
  
  v = np.array(s["Parameters"])
  # print(np.shape(v))
  dim = len(v)
  # cov = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
  cov = np.identity(dim, dtype=float)
  cov_inv = np.linalg.inv(cov)

  mu = np.zeros(dim, dtype=float)
  v_centred = v - mu
  # print(np.shape(covariance))
  # s["P(x)"] = -0.5 * (v1 * v1)
  s["P(x)"] = -0.5 * np.matmul(np.matmul(v_centred.T, cov_inv), v_centred)