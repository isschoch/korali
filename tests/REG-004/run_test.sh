#!/bin/bash

source ../functions.sh

#################################################
# Execute Solver Scripts
#################################################

echo "[Korali] Beginning solver tests"

for file in *.py
do
  echo "-------------------------------------"
  echo "Running File: ${file%.*}"
  
  python3 ./$file
  check_result

  echo "[Korali] Removing results..."
  rm -rf "_korali_result" 
  check_result

  echo "-------------------------------------"
done

