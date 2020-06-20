#!/usr/bin/env bash
# File       : testDefault.sh
# Created    : Fri Mar 20 2020 09:18:08 AM (+0100)
# Author     : Fabian Wermelinger
# Description: Tests
# Copyright 2020 ETH Zurich. All Rights Reserved.
set -e

ID='TEST'
nThreads=12
nSamples=3000
confidence=0.95
popSize=8672489
fDays=2
data='1.0,8.0,8.0,18.0,27.0,42.0,56.0,90.0,114.0,214.0,268.0,337.0,374.0,491.0,652.0,652.0,1139.0,1359.0,2200.0,2200.0,2700.0,3028.0,4075.0'
data="${data//,/ }"

resultPath=/tmp/request$ID
rm -rf ${resultPath}
mkdir -p ${resultPath}
cp *.py *.json $resultPath
pushd $resultPath

python3 ./sample.py --nThreads $nThreads --silent \
    --data $data \
    --populationSize $popSize \
    --nSamples $nSamples

python3 ./propagate.py --nThreads $nThreads --silent \
    --data $data \
    --populationSize $popSize \
    --futureDays $fDays

python3 ./intervals.py \
    --data $data \
    --populationSize $popSize \
    --nSamples $nSamples \
    --percentages $confidence

mv intervals.json output${ID}.json

popd

mv ${resultPath} .
