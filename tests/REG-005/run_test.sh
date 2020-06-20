#!/bin/bash

source ../functions.sh

pushd ../../

echo "[Korali] Beginning profiling tests..."

profFiles=`find . -name profiling.json`

for f in $profFiles
do
   echo "----------------------------------------------"
   echo " Processing profiler information from $f ..."
   echo "----------------------------------------------"
   echo "python3 -m korali.profiler --test --input $f "
   check_result
done

popd
