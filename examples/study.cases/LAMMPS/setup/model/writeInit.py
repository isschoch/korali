#!/usr/bin/python
import numpy as np
import scipy
import sys
import os.path
from math import pow
from operator import itemgetter # for sorting a dict

M = int(float(sys.argv[1]))
trj = sys.argv[2]

#########################################################################
if (os.path.exists(trj)==False): 
    print "--1"
    exit()
#########################################################################
readBox = False
readCoord = False
readFirst = True
coordsAll = []
f = open(trj)
box = []
for line in f:
	if "ITEM: ATOMS" in line:
	    readBox = False
	    readFirst = False
	    readCoord = True
	elif "TIMESTEP" in line:
	    readCoord = False
	    if (readFirst==False): coordsAll.append(sorted(coords, key=itemgetter(0)))
	    coords = []
	elif "BOX" in line:
	    readBox = True
	    box = []
	elif readBox == True:
	    box.append( map(float, line.split()) )
	elif readCoord == True:
	    coords.append( map(float, line.split()) )
coordsAll.append(sorted(coords, key=itemgetter(0)))
   
delta = [box[0][0],box[1][0],box[2][0]] 
size = [box[0][1]-box[0][0],box[1][1]-box[1][0],box[2][1]-box[2][0]]

#########################################################################
vol = size[0]*size[1]*size[2]/pow(10,24)
nMolecules = len(coordsAll[len(coordsAll)-1])
dens = nMolecules*M*18.0153/6.0221415/pow(10,23)/vol
    

if (dens<0.85 or dens>1.15):
    print "-1"
    exit()
#########################################################################

coordNew = []
coordLast = coordsAll[len(coordsAll)-1]
for kk in range(0,len(coordLast),1):
	tmp1 = [coordLast[kk][2]-delta[0],coordLast[kk][3]-delta[1],coordLast[kk][4]-delta[2]]
	for dim in range(3):
	    if (tmp1[dim]<0.0): tmp1[dim] += size[dim]
	    elif (tmp1[dim]>size[dim]): tmp1[dim] -= size[dim]
	coordNew.append(tmp1)

delta = [0.0,0.0,0.0] 
#########################################################################
pdb = open("initial.txt", 'w')

pdb.write("LAMMPS 'data.' description \n")
pdb.write("\n")
pdb.write("      %d atoms\n"% (len(coords)))
pdb.write("\n")
pdb.write("       1 atom types\n")
pdb.write("\n")
pdb.write("    %1.2f %1.2f      xlo xhi\n" % (0.0,size[0]) )
pdb.write("    %1.2f %1.2f      ylo yhi\n" % (0.0,size[1]) )
pdb.write("    %1.2f %1.2f      zlo zhi\n" % (0.0,size[2]) )
pdb.write("\n\n")
pdb.write("Atoms\n")
pdb.write("\n")

natom = 1
nmol = 1
for kk in range(0,len(coordNew),1):
	tmp1 = [coordNew[kk][0]-delta[0],coordNew[kk][1]-delta[1],coordNew[kk][2]-delta[2]]
	pdb.write("     %d   %d  1  %1.10f   %1.4f    %1.4f    %1.4f\n" % (natom,nmol,0.0,tmp1[0],tmp1[1],tmp1[2]) )
	natom += 1
	nmol += 1

pdb.write("\n\n")
pdb.write("Velocities\n")
pdb.write("\n")

natom = 1
for kk in range(0,len(coordLast),1):
	pdb.write("     %d   %1.8f    %1.8f    %1.8f\n" % (natom,coordLast[kk][5],coordLast[kk][6],coordLast[kk][7]) )
	natom += 1

pdb.close()
