#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../../tests/functions.sh

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _korali_result* _executor_output; check_result

##### Running Tests 

python3 ./run-execution.py; check_result
  
