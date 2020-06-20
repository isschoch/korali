#!/bin/bash

###### Auxiliar Functions and Variables #########

source ../tests/functions.sh 


### Testing examples

exampleDirs=`find -type d -not -path "*/_*" -not -path "*/study.cases/*" -not -name "study.cases" | sort | awk '$0 !~ last "/" {print last} {last=$0} END {print last}'`

for dir in $exampleDirs
do
  echo " + Entering Folder: $dir"
  pushd $dir; check_result
   
  ./.run_test.sh; check_result
  
  popd; check_result
done

### Testing Study Cases

exampleDirs=`ls -d study.cases/*/`

for dir in $exampleDirs
do
  echo " + Entering Folder: $dir"
  pushd $dir; check_result
  
  ./.run_test.sh; check_result
  
  popd; check_result
done

exit 0 

