#!/bin/bash

###### Auxiliar Functions and Variables #########

function check_result()
{
 if [ ! $? -eq 0 ]
 then 
  echo "[Korali] Error detected."
  exit -1
 fi 
}

##### Creating test files

echo "  + Creating test files..."

rm -f _*; check_result
fileList=`ls *.py`

for file in $fileList
do
 sed -e 's/k.resume(/k["Dry Run"] = True; k.resume(/g'\
     -e 's/k.run(/k["Dry Run"] = True; k.run(/g'\
  $file > _$file; check_result
done

##### Running Tests

echo "  + Running test files..."

for file in $fileList
do
 python3  _$file; check_result
done

##### Deleting Tests

echo "  + Removing test files..."

rm -f _run*; check_result

