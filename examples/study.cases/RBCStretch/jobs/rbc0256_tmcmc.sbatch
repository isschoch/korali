#!/bin/bash -l
#SBATCH --job-name="rbc0256_tmcmc"
#SBATCH --output=rbc0256_tmcmc.%j.out
#SBATCH --time=1:00:00
#SBATCH --nodes=257
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

date
mir.run --runargs "-n 257 --ntasks-per-node=1" ./run-tmcmc.py 256
date
