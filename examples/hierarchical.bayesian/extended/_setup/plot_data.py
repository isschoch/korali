#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

for k in range(5):
  dataFile = "../setup/data/data_set_" + str(k).zfill(3) + ".dat"
  x = np.loadtxt(dataFile)
  plt.plot(x[:, 0], x[:, 1], '-o')

plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('time')
plt.ylabel('noisy logistic data')
plt.title('All data sets')
plt.grid(True)
plt.show()
