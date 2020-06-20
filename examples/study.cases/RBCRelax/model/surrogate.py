import os
import sys
from os import path

# Extract timesteps
def getReferencePoints(expName):
  time = []
  fname = './data/' + expName + '.txt'
  for line in open(fname, 'r'):
    tmp = line.split(" ")
    time.append(float(tmp[0]))

  return time

# Extract RBC param
def getReferenceData(expName):
  data = []
  fname = './data/' + expName + '.txt'
  for line in open(fname, 'r'):
    tmp = line.split(" ")
    data.append(float(tmp[1]))

  return data
 
def surrogate_henon(gammaC, LLmax, time):
  import numpy as np
  a = 9.9595100e-06
  b = 7.2304811e-03
  tc = a*gammaC + b
  EXP_out = np.exp(-time/tc)
  
  LLinf = 1.
  Const = (LLmax+LLinf)/(LLmax-LLinf)
  LL0 = LLinf * (-EXP_out-Const) / (EXP_out-Const)
  return LL0


def surrogate_hochmuth(gammaC, LWmax, time):
  import numpy as np
  a = 0.35453265
  b = 1.12145917
  tc = (np.exp(-gammaC*1.e-4/a) + b) * gammaC * 1.e-5
  EXP_out = np.exp(-time/tc)
  
  LWinf = 1.
  Const = (LWmax+LWinf)/(LWmax-LWinf)
  LW = LWinf * (-EXP_out-Const) / (EXP_out-Const)
  return LW
  

def relaxModel_surrogate(s, x, expName, tend ):
  import numpy as np
  sid    = s["Sample Id"]
  curGen = s["Current Generation"]
  gammaC = s["Parameters"][0]
  
  NDATA = len(x)

  time = np.linspace(0, tend, NDATA)

  quants = []
  if expName == "henon":
    LLmax = 1.065484075333
    quants = surrogate_henon(gammaC, LLmax, time)
  elif expName == "hochmuth01" :
    LWmax = 2.0045070422535214
    quants = surrogate_hochmuth(gammaC, LWmax, time)
  elif expName == "hochmuth02" :
    LWmax = 2.1898454000168965
    quants = surrogate_hochmuth(gammaC, LWmax, time)
  elif expName == "hochmuth03" :
    LWmax = 1.941251354564488
    quants = surrogate_hochmuth(gammaC, LWmax, time)
  elif expName == "hochmuth04" :
    LWmax = 2.0045111517521175
    quants = surrogate_hochmuth(gammaC, LWmax, time)
  else :
    sys.exit("expName ({0}) nor recognized!".format(expName))

  s["Reference Evaluations"] = quants.tolist()
