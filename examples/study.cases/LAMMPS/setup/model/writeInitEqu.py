#!/usr/bin/python
import numpy as np
import scipy
import sys
import random
import time
from math import pi ,sqrt, cos, sin

random.seed(time.time())

M = int(float(sys.argv[1]))

nrepeat = int(sys.argv[2])
nMol = nrepeat*nrepeat*nrepeat
nAtoms = nMol

d = [0.0,3.11,4.0,4.48,4.93,5.31,5.65]
d0 = d[M]

pdb = open("initialEqu.txt", 'w')

pdb.write("LAMMPS 'data.' description \n")
pdb.write("\n")
pdb.write("      %d atoms\n"% (nAtoms))
pdb.write("\n")
pdb.write("       1 atom types\n")
pdb.write("\n")
pdb.write("    0.0 %1.2f      xlo xhi\n" % (d0*nrepeat) )
pdb.write("    0.0 %1.2f      ylo yhi\n" % (d0*nrepeat) )
pdb.write("    0.0 %1.2f      zlo zhi\n" % (d0*nrepeat) )
pdb.write("\n\n")
pdb.write("Atoms\n")
pdb.write("\n")

natom = 1
nmol = 1
for kx in range(nrepeat):
    for ky in range(nrepeat):
	for kz in range(nrepeat):
	    coord = [d0/2.0+kx*d0,d0/2.0+ky*d0,d0/2.0+kz*d0]
	    pdb.write("     %d   %d  1  %1.10f   %1.4f    %1.4f    %1.4f\n" % (natom,nmol,0.0,coord[0],coord[1],coord[2]) )
	    natom += 1
	    nmol += 1

 
pdb.close()







