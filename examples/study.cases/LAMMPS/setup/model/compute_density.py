#!/usr/bin/python
import numpy as np
import scipy
import sys
import os.path
from math import pow

Tstr = sys.argv[1]
T = float(Tstr)

if (os.path.exists("./density.txt")): 

    density = []
    lines = open("./density.txt")
    for line in lines:
        if "#" in line: continue
	l = map(float, (line).split())
	density.append(l[1])  

    if (len(density)>100):
    	naverage = 100 	#last 100
    	dens = 0.0
    	for kk in range(len(density)-naverage,len(density)):
            dens += density[kk]

    	dens /= naverage 
    else:
    	dens = -1.0

else:
    dens = -1.0


print dens
