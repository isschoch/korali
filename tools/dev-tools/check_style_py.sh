#!/usr/bin/env bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error fixing style." 
  exit -1
 fi 
}

fileDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pushd $fileDir

root=$fileDir/../..

##############################################
### Testing Python Code Style
##############################################

PIP_USER=$(python3 -c "import sys; hasattr(sys, 'real_prefix') or print('--user')")

python3 -m yapf --version > /dev/null
if [ $? -ne 0 ]; then

  echo "[Korali] yapf not found, trying to install it automatically."
  python3 -m pip install $PIP_USER yapf; check
fi

src_files=`find $root -type f -not -name "__*"  -name "*.py" \
          -not -path "${root}/tools/dev-tools/*" \
          -not -path "${root}/source/external/*" \
          -not -path "${root}/external/*" \
          -not -path "${root}/tutorials/examples/*"`
          
diff=`echo $src_files | xargs -n6 -P2 python3 -m yapf --style=yapf -d "$@"`

if [ ! "$diff" == "" ]; then
 echo "[Korali] Error: Python Code formatting is not normalized:"
 echo $diff | head -n 5
 echo "[Korali] Solution: Please run $fileDir/fix_style_py.sh to fix it."
 exit -1
else
 echo "[Korali] Python code formatting is correct."
fi

popd

exit 0
