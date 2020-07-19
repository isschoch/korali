import numpy as np


def compareMean(k):
  samples = k["Solver"]["Sample Database"]

  mean = np.mean(samples)
  chainmean = k["Solver"]["Chain Mean"]

  assert np.isclose(mean, chainmean), "Chain Mean deviates from Mean of "\
          "Samples ({0} vs {1})".format(mean, chainmean)


def compareStd(k):
  samples = k["Solver"]["Sample Database"]

  mean = np.mean(samples)
  std = np.sqrt(sum((samples - mean)**2) / (len(samples) - 1))
  chainstd = k["Solver"]["Cholesky Decomposition Chain Covariance"]

  assert np.isclose(std, chainstd), "Cholesky Decomposition of Chain" \
          "Covariance deviates from Standard Deviation of Samples ({0} vs {1})".format(std, chainstd)


def compareMeanHMC(k):
  warmupSamples = k["Solver"]["Sample Database"]

  mean = np.mean(warmupSamples)
  positionMean = k["Solver"]["Position Mean"]

  assert np.isclose(mean, positionMean), "Position Mean deviates from Mean of "\
          "Samples ({0} vs {1})".format(mean, positionMean)


def compareStdHMC(k):
  warmupSamples = k["Solver"]["Warmup Sample Database"]

  mean = np.mean(warmupSamples)
  std = np.sqrt(sum((warmupSamples - mean)**2) / (len(warmupSamples) - 1))
  inverseMetricSqrt = np.sqrt(k["Solver"]["Inverse Metric"])

  assert np.isclose(std, inverseMetricSqrt), "Inverse Metric Sqrt" \
          "Inverse Metric Sqrt deviates from Standard Deviation of Samples ({0} vs {1})".format(std, inverseMetricSqrt)
