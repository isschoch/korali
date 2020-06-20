#!/bin/bash -l
#SBATCH --job-name="cmaes_multiple"
#SBATCH --output="cmaes_multiple.%j.out"
#SBATCH --time=24:00:00
#SBATCH --nodes=33
#SBATCH --ntasks-per-node=2
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

set -eu

date
cd ..
pushd results; ./backup.sh; popd;
nranks=`expr $SLURM_JOB_NUM_NODES \* 2 - 1`
echo "Running Korali+Mirheo on $nranks ranks"
mir.run --runargs "-n $nranks" ./run-cmaes-multiple.py
date
