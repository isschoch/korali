#!/bin/bash

# Brief: Re-run all examples and features for basic sanity check.

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

dir=$PWD/../../examples

echo "[Korali] Beginning examples test..."
echo "-------------------------------------"

pushd $dir; check_result

./.run_test.sh
check_result

popd
check_result

echo "-------------------------------------"

############# STEP 2 ##############

dir=$PWD/../../features

echo "[Korali] Beginning features test..."
echo "-------------------------------------"

pushd $dir; check_result

./.run_test.sh
check_result

popd
check_result

echo "-------------------------------------"
