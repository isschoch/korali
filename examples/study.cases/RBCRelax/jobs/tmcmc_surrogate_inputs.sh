#!/bin/bash -l
set -eu

date
cd ../hierarchical/
echo "Running Korali+Surrogate ..."

#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.004 --burnin 10 
#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.006 --burnin 10 
#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.008 --burnin 10 
#./phase2_surrogate_inputs.py --ub1 10.0 --ub2 5.0 --beta2 0.01 --burnin 9
#./phase2_surrogate_inputs.py --ub1 10.0 --ub2 5.0 --beta2 0.04 --burnin 9
#./phase2_surrogate_inputs.py --ub1 10.0 --ub2 5.0 --beta2 0.08 --burnin 9

./phase2_surrogate_inputs.py --ub1 15.0 --ub2 5.0 --beta2 0.01 --burnin 3 &
./phase2_surrogate_inputs.py --ub1 15.0 --ub2 5.0 --beta2 0.01 --burnin 5 &
./phase2_surrogate_inputs.py --ub1 15.0 --ub2 5.0 --beta2 0.01 --burnin 7 &
./phase2_surrogate_inputs.py --ub1 15.0 --ub2 5.0 --beta2 0.008 --burnin 3 &
./phase2_surrogate_inputs.py --ub1 15.0 --ub2 5.0 --beta2 0.008 --burnin 5 &
./phase2_surrogate_inputs.py --ub1 15.0 --ub2 5.0 --beta2 0.02 --burnin 7 &
#
#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.02 --burnin 10 
#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.04 --burnin 10
#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.08 --burnin 10
#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.16 --burnin 10
#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.24 --burnin 10
#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.32 --burnin 10
#./phase2_surrogate_inputs.py --ub 10.0 --beta2 0.40 --burnin 10
date
