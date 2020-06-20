#!/bin/bash

###### Auxiliar Functions and Variables #########

function check_result()
{
 if [ ! $? -eq 0 ]
 then 
  echo "[Korali] Error detected."
  exit -1
 fi 
}

##### Deleting Previous Results

echo "  + Deleting previous results..."
rm -rf _results_*; check_result
 
##### Running Tests

echo "  + Running test files..."

python3 ./run-example1.py; check_result
python3 ./run-example2.py; check_result
  
