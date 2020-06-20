#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result*; check_result
 
##### Creating test files

echo "  + Creating test files..."

##### Running Tests

python3 ./run-adam.py; check_result
python3 ./run-rprop.py; check_result

  
