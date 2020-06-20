#!/bin/bash -l
#SBATCH --job-name="rbc4096_tmcmc"
#SBATCH --output=rbc4096_tmcmc.%j.out
#SBATCH --time=0:40:00
#SBATCH --nodes=4097
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=large
#SBATCH --constraint=gpu

cd ..

date
mir.run --runargs "-n 4097 --ntasks-per-node=1" ./run-tmcmc.py 4096
date
