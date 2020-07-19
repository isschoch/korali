#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..."  
rm -rf _korali_result*; check_result 
 
##### Running Tests

python3 ./run-surrogates.py; check_result

