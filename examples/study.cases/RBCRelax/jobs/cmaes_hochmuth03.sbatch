#!/bin/bash -l
#SBATCH --job-name="cmaes_hochmuth03"
#SBATCH --output="cmaes_hochmuth03.%j.out"
#SBATCH --time=24:00:00
#SBATCH --nodes=9
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
mir.run --runargs "-n $nranks" ./run-cmaes.py --exp hochmuth03 --lower 8000 --upper 32000 --tend 0.4 --inimesh_fname "stretch_H1979_d03.off"
date
