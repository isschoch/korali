#!/usr/bin/env python3
import os
import sys
import numpy as np


def prepareOutputDir():
  if os.path.isdir('_executor_output') == False:
    os.mkdir('_executor_output')


def put_normal_rnds(theta, Ns, fileName):
  mu = theta["Parameters"][0]
  var = theta["Parameters"][1]

  y = np.random.normal(mu, var, Ns)

  if os.path.isdir('_executor_output') == True:
    f = open('_executor_output/{0}'.format(fileName), 'a+')
    np.savetxt(f, np.transpose(y))
    f.close()

  else:
    sys.exit('put_normal_rnds: dir \'_executor_output\' does not exist! exit..')
