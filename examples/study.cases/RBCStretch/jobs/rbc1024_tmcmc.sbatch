#!/bin/bash -l
#SBATCH --job-name="rbc1024_tmcmc"
#SBATCH --output=rbc1024_tmcmc.%j.out
#SBATCH --time=0:30:00
#SBATCH --nodes=1025
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..

date
mir.run --runargs "-n 1025 --ntasks-per-node=1" ./run-tmcmc.py 1024
date
