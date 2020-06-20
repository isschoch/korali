#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../tests/functions.sh

###### Checking if MPI available ##########

if [[ $MPICXX == "" ]]
then
 echo "[Korali] MPI not installed, skipping test."
 exit 0
fi

##### Deleting Previous Results

echo "  + Deleting previous results..." 
rm -rf _korali_result*; check_result

##### Recompiling C++

make clean; check_result
make -j4; check_result

##### Running Tests

mpirun -n 9 ./run-cmaes; check_result
mpirun -n 9 ./run-tmcmc; check_result

