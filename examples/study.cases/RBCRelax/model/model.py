import os
import sys
import csv
import shutil
from os import path

sys.path.append('../post')

from simulation import run_korali
from compute_extensions import process_LW, process_LL0

# Dump Folder for temporary ply files
scratch = './scratch/'
jid = 0

if ('SLURM_JOBID' in os.environ): jid  = os.environ['SLURM_JOBID']
if ('SCRATCH' in os.environ): scratch  = os.environ['SCRATCH']

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

# Find and return file list
def getPlyFiles(dirname):
  ply_files = os.listdir(dirname)
  ply_files = [f for f in ply_files if f.endswith('.ply')]
  ply_files = [ dirname + fn for fn in ply_files ]
  ply_files.sort()
  if not ply_files: sys.exit("Model Error: No Ply files found in " + dirname + "!")
  return ply_files

def relaxModel(s, x, expName, tend, ini_mesh_fname, comm_address ):
  
  sid    = s["Sample Id"]
  curGen = s["Current Generation"]
  gammaC = s["Parameters"][0]
  
  NDATA = len(x)

  ref_mesh_fname = "./data/off_files/reference.off" 
  eq_mesh_fname  = "./data/off_files/equilibrium.off"
  
  ply_dir = "{0}/korali/job{1}/{2}/gen{3}/sample{4}/".format(scratch,jid,expName,curGen,sid)

  #Creating Directory
 
  createdDir = False
  while (not createdDir):
   try:
    createdDir = True
    if not os.path.exists(ply_dir): os.makedirs(ply_dir) 
   except OSError as e:
    if e.errno != os.errno.EEXIST: createdDir = False
    pass

  run_korali( comm_address, gammaC, NDATA, tend, ini_mesh_fname, ref_mesh_fname, ply_dir )

  ply_files = getPlyFiles(ply_dir)

  quants = []
  if expName == "henon":
    quants    = process_LL0(ply_files, eq_mesh_fname).tolist()
  else: # Hochmuth
    quants    = process_LW(ply_files, eq_mesh_fname).tolist()
  
  s["Reference Evaluations"] = quants
