#!/bin/bash -l
#SBATCH --job-name="rbc0004_tmcmc"
#SBATCH --output=rbc0004_tmcmc.%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

date
mir.run --runargs "-n 5 --ntasks-per-node=1" ./run-tmcmc.py 4
date
