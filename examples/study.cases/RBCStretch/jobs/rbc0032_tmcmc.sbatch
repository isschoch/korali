#!/bin/bash -l
#SBATCH --job-name="rbc0032_tmcmc"
#SBATCH --output=rbc0032_tmcmc.%j.out
#SBATCH --time=1:00:00
#SBATCH --nodes=33
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

date
mir.run --runargs "-n 33 --ntasks-per-node=1" ./run-tmcmc.py 32
date
