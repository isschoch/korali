#!/bin/bash -l
#SBATCH --job-name="rbc2048_tmcmc"
#SBATCH --output=rbc2048_tmcmc.%j.out
#SBATCH --time=0:30:00
#SBATCH --nodes=2049
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

date
mir.run --runargs "-n 2049 --ntasks-per-node=1" ./run-tmcmc.py 2048
date
