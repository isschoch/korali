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
rm -rf _korali_result*; check_result
 
##### Creating test files

echo "  + Creating test files..."

sed 's/k.run(/k["Dry Run"] = True; k.run(/g' run-cmaes.py > __test-cmaes.py
sed 's/k.run(/k["Dry Run"] = True; k.run(/g' run-tmcmc.py > __test-tmcmc.py

##### Running Tests

echo "  + Running test files..."

python3 ./__test-cmaes.py; check_result
python3 ./__test-tmcmc.py; check_result

##### Deleting Tests

echo "  + Removing test files..."

rm __test*

  
