#!/bin/bash

source ../functions.sh


#################################################
# MIN/MAX Optimizer
#################################################

echo "[Korali] Beginning minimizing mazimizing tests"

echo "-------------------------------------"
echo "Testing Maximizing"
echo "Running File: run-maxcmaes1.py"

python3 ./run-maxcmaes1.py 
check_result

echo "[Korali] Removing results..."
rm -rf "_korali_result" 
check_result

echo "-------------------------------------"

echo "-------------------------------------"
echo "Testing Maximizing"
echo "Running File: run-maxcmaes2.py"

python3 ./run-maxcmaes2.py 
check_result

echo "[Korali] Removing results..."
rm -rf "_korali_result" 
check_result

echo "-------------------------------------"

echo "-------------------------------------"
echo "Testing Minimizing"
echo "Running File: run-mincmaes1.py"

python3 ./run-mincmaes1.py 
check_result

echo "[Korali] Removing results..."
rm -rf "_korali_result" 
check_result

echo "-------------------------------------"

echo "-------------------------------------"
echo "Testing Minimizing"
echo "Running File: run-mincmaes2.py"

python3 ./run-mincmaes2.py 
check_result

echo "[Korali] Removing results..."
rm -rf "_korali_result" 
check_result

echo "-------------------------------------"

echo "-------------------------------------"
echo "Testing Minimizing C++"
echo "Build & Running File: run-mincmaes1.cpp"

make
python3 ./run-mincmaes2.py 
check_result

echo "[Korali] Removing results..."
rm -rf "_korali_result" 
check_result

make clean
echo "-------------------------------------"
