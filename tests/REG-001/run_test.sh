#!/bin/bash

# Check Code Formatting and documentation

###### Auxiliar Functions and Variables #########
 
source ../functions.sh

############# STEP 1 ##############

echo "[Korali] Beginning code formatting check..."

pushd ../../tools/dev-tools/
check_result

 ./check_style_cxx.sh 
 check_result

popd
check_result

############## STEP 2 ################

echo "[Korali] Beginning code formatting check..."

pushd ../../docs/
check_result

 ./build.sh
 check_result

popd
check_result
