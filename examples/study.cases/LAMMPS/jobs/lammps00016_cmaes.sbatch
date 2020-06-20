#!/bin/bash -l
#SBATCH --job-name="lammps0016_cmaes"
#SBATCH --output=lammps0016_cmaes.%j.out
#SBATCH --time=03:00:00
#SBATCH --nodes=16
#SBATCH --account=s929
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

date
./run-cmaes.py 
date
