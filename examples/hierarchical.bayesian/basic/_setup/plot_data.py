#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

for k in range(5):
  dataFile = "../_setup/data/data_set_" + str(k).zfill(3) + ".dat"
  x = np.loadtxt(dataFile)
  count, bins, ignored = plt.hist(x, 'sturges', density=True, rwidth=0.9)

plt.autoscale(enable=True, axis='x', tight=True)
plt.title('Histogram of all data sets')
plt.grid(True)
plt.show()
