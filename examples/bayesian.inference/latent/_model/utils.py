import numpy as np


def multivariate_gaussian_probability(mus, nDimensions: int, assignments,
                                      nClusters: int, sigma: float, points):
  # /* Normalized, no log, clean pretty probability for N points. */
  assert (sigma > 0.0)
  assert (nDimensions > 0)
  for a in assignments:
    assert ((0 <= a) and (a < nClusters))
  for v in points:
    assert (len(v) == nDimensions)

  N = len(points)

  p = 1.0
  for i in range(N):
    # get the cluster mean
    c = assignments[i]
    mu = mus[c]
    # update p
    p = p * univariate_gaussian_probability(mu, sigma, points[i])
  return p


def univariate_gaussian_probability(mu, sigma: float, point):
  point = np.array(point)
  mu = np.array(mu)
  n = len(mu)
  assert (len(mu) == len(point))
  distance = mu - point
  squared_distance = np.inner(distance, distance)
  exponent = -squared_distance / (2.0 * sigma**2)
  denominator = (np.sqrt(2.0 * np.pi) * sigma)**n
  probability = np.exp(exponent) / denominator
  return probability
