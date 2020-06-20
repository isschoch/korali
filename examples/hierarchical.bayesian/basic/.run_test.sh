#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../tests/functions.sh

##### Deleting Previous Results
 
echo "  + Deleting previous results..."

pushd _setup; check_result
./clean.sh; check_result
popd; check_result

##### Running Tests

python3 ./phase0.py; check_result
python3 ./phase1.py; check_result
python3 ./phase2.py; check_result
python3 ./phase3a.py; check_result
python3 ./phase3b.py; check_result

