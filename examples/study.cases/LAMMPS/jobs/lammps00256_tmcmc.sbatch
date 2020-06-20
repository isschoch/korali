#!/bin/bash -l
#SBATCH --job-name="lammps0256_tmcmc"
#SBATCH --output=lammps0256_tmcmc.%j.out
#SBATCH --error=lammps0256_tmcmc.%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=256
#SBATCH --account=s929
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

date
./run-tmcmc.py 256
date
