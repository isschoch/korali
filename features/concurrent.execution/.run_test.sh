#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..." 
rm -rf _korali_result*; check_result

##### Running Tests

python3 ./run.py 1; check_result
python3 ./run.py 2; check_result
python3 ./run.py 4; check_result
python3 ./run.py 8; check_result

