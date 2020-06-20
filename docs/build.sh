#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error building site."
  exit -1
 fi 
}

# Building Korali source
pushd ../source
check

python3 ./build.py
check

popd
check

# Installing shpinx, mkdocs, and materials theme
python3 -m pip install sphinx --user
check

python3 -m pip install sphinx_rtd_theme --user
check

python3 -m pip install Pygments --user
check

# Getting korali-apps submodule
pushd ..
git submodule update --init --recursive --remote
popd

# Building User Manual
pushd manual
check

pushd builder
check

python3 ./buildExamples.py 
check

python3 ./buildFeatures.py 
check

python3 ./buildTests.py 
check

python3 ./buildModules.py 
check

python3 ./buildTools.py
check

popd
check

make html
check

popd

# Inserting user manual into website

mkdir -p web/docs
check 

cp -r manual/.build/html/* web/docs
check

doxygenBin=../external/doxygen
# If doxygen is not installed, run the installation script
if [ ! -f $doxygenBin ]; then
 pushd ..
 ./external/install_doxygen.sh
 popd
fi

# Running Doxygen
echo "Using $doxygenBin for C++ documentation..."
$doxygenBin doxygen.config 2>&1 | grep -E 'warning|error'
if [ $? -eq 0 ]; then
 echo "[Korali] Error running doxygen."
 echo "[Korali] Hint: Check if there is any missing variable/function documentation."
 echo "[Korali] Hint: Also make sure doxygen is correctly installed by running:"
 echo "[Korali]           > {KORALI_ROOT}/external/install_doxygen.sh"
 exit -1
fi 

echo "[Korali] Webpage Build complete."
