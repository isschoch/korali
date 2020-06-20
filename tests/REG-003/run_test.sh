#!/bin/bash

source ../functions.sh

pushd ../../examples/; check_result

echo "[Korali] Beginning plotting tests"

resDirs=`find . -name "*_result*"`

for dir in $resDirs
do
  echo "-------------------------------------"
  echo " Plotting results from $dir ..."
  echo "-------------------------------------"
  python3 -m korali.plotter --test --dir "${dir}" 
  check_result

done

popd
