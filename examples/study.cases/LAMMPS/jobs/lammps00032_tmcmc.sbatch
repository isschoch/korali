#!/bin/bash -l
#SBATCH --job-name="lammps0032_tmcmc"
#SBATCH --output=lammps0032_tmcmc.%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=32
#SBATCH --account=s929
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

date
./run-tmcmc.py 32
date
