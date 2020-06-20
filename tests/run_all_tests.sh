#!/bin/bash

source functions.sh

echo "" > $logFile

#Running tutorials

for dir in */
do
  echo "[Korali] Running test in folder ${dir} ..."
  pushd $dir
  # Keep original fail code if something goes wrong
  set -o pipefail
  ./run_test.sh 2>&1 | tee $logFile

  if [ ! $? -eq 0 ]; then
   echo ""
   echo "-------------------------------------------------"
   echo "[Korali] Tests failed at folder ${dir}"
   echo "[Korali] see tests/test.log for more information"
   echo "-------------------------------------------------"
   exit -1
  fi

  popd
done

echo "[Korali] All tests ran correctly."
