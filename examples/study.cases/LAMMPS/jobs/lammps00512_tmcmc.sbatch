#!/bin/bash -l
#SBATCH --job-name="lammps0512_tmcmc"
#SBATCH --output=lammps0512_tmcmc.%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=512
#SBATCH --account=s929
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

date
./run-tmcmc.py 512
date
