#!/bin/bash

source ../functions.sh

echo "[Korali] Beginning Stat Test 002"

for file in run*.py
do
  echo "-------------------------------------"
  echo " Running $file"
  echo "-------------------------------------"
  ./"$file" 
  check_result
done

echo "[Korali] Removing results..."
rm -rf _result_run-* 
check_result
