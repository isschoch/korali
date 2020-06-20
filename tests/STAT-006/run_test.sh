#!/bin/bash

source ../functions.sh

constraints=(
"None"
"Inactive"
"Active at Max 1"
"Active at Max 2"
"Inactive at Max 1"
"Inactive at Max 1"
"Mixed"
"Stress"
)

#################################################
# CCMA-ES Termination Criterion Tests
#################################################

echo "[Korali] Beginning CMA-ES termination criterion tests"

for c in "${constraints[@]}"
do

  echo "-------------------------------------"
  echo "Testing Constraints: ${c}"
  echo "Running File: run-ccmaes.py"

  python3 ./run-ccmaes.py --constraint "${c}"
  check_result

  echo "[Korali] Removing results..."
  rm -rf "_korali_result"
  check_result

  echo "-------------------------------------"

done
