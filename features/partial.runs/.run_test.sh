#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..." 
rm -rf _korali_result*; check_result

##### Running Tests

python3 ./run-cmaes.py; check_result
python3 ./run-dea.py; check_result
python3 ./run-rprop.py; check_result
python3 ./run-propagation.py; check_result
python3 ./run-mcmc.py; check_result
python3 ./run-tmcmc.py; check_result
python3 ./run-multiple.py; check_result

