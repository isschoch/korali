#!/usr/bin/env python
import os
import numpy as np

# Load Forces for experiments and the expected (measured) diameters (low & high)
def getReferenceData():
    this_dir, this_filename = os.path.split(__file__)
    data_dir = os.path.join(this_dir, "data")
    data_fname = os.path.join(data_dir, "Mills.2004.txt")   
    data = np.loadtxt(data_fname)
    diam_lo_exp = data[:,1]
    diam_hi_exp = data[:,2]
    diam_hi_lo = np.append(diam_lo_exp, diam_hi_exp)
    return diam_hi_lo
    
def getReferencePoints():
    this_dir, this_filename = os.path.split(__file__)
    data_dir = os.path.join(this_dir, "data")
    data_fname = os.path.join(data_dir, "Mills.2004.txt")
    data = np.loadtxt(data_fname)
    forces = data[:,0]
    return forces    

