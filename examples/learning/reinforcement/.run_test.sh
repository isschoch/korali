#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results 

echo "  + Deleting previous results..."
rm -rf _korali_result*; check_result

##### Creating test files
 
echo "  + Creating test files..."

sed -e 's%Defining Termination Criteria%Defining Termination Criteria\ne["Solver"]["Termination Criteria"]["Max Generations"] = 20\n%g' \
        run-dqn.py > __test-dqn.py; check_result

##### Running Test

OMP_NUM_THREADS=4 python3 ./__test-dqn.py; check_result

##### Deleting Tests

rm -rf __test-*; check_result
  
  
