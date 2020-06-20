#!/bin/bash -l
#SBATCH --job-name="rbc0064_cmaes"
#SBATCH --output=rbc0064_cmaes.%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=65
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ..

date
mir.run --runargs "-n 65 --ntasks-per-node=1" ./run-cmaes.py 64
date
