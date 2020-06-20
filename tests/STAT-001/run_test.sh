#!/bin/bash

source ../functions.sh

############# STEP 1 ##############

echo "[Korali] Beginning sampling statistics tests..."

for file in run*.py
do
  echo "-------------------------------------"
  echo " Running $file"
  echo "-------------------------------------"
  ./"$file" >> $logFile
  check_result
done

echo "[Korali] Removing results..."
rm -rf _result_run-* 
check_result
