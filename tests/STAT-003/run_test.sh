#!/bin/bash

source ../functions.sh

############# STEP 1 ##############

echo "[Korali] Beginning STAT-003 test"
echo "[Korali] minimization with non-finites tests..."

for file in run*.py
do
  echo "-------------------------------------"
  echo " Running $file"
  echo "-------------------------------------"
  ./"$file" 
  check_result
done

echo "[Korali] Removing results..."
rm -rf _results_* 


