#!/bin/bash

######### Global Definitions ########
libName="libco"
minVersion=

######### Helper Functions ########

function check()
{
 if [ ! $? -eq 0 ]
 then
  exit $?
 fi 
}

# Help display function
function print_help ()
{
 echo ""
 echo "Syntax: ./install_${libName}.sh [--jobs=N] [--check]"
 echo ""
 echo "Where:"
 echo " --no-install If ${libName} is not found, do not download and install it."
 echo " --jobs=N Specifies N jobs to use when building ${libName}."
 echo " --help Displays this help message."
}

######### Environment Configuration ########

NJOBS=4
baseKoraliDir=$PWD
foundbinVersionFile=0
install=1

if [ -f $baseKoraliDir/docs/VERSION ]; then
 foundbinVersionFile=1
fi

if [ -f $baseKoraliDir/../docs/VERSION ]; then
 foundbinVersionFile=1
 baseKoraliDir=`dirname $PWD`
fi

if [ $foundbinVersionFile == 0 ]; then
  echo "[Korali] Error: You need to run this file from Korali's base folder."
  exit 1
fi

######### Argument Parsing ########

for i in "$@"
do
case $i in
    --jobs=*)
    NJOBS="${i#*=}"
    shift
    ;;
    --no-install)
    install=0
    shift 
    ;;
    --help)
    print_help
    exit 0 
    shift 
    ;;
    *)
    print_help
    echo ""
    echo "[Korali] Error:  Unknown option $i."       
    exit 1
    ;;
esac
done

######## Checking for existing software ########

externalDir=${baseKoraliDir}/external

baseLibDir=${externalDir}/${libName}
installDir=${baseLibDir}/install/
buildDir=${baseLibDir}/build

fileFound=0
localFile=${installDir}/libco.o

if [ -f ${localFile} ]; then
 fileFound=1
 filePath=${localFile}
fi
 
######## If not installed, download and install ########

if [ ${fileFound} == 0 ]; then

 if [ ${install} == 0 ]; then
   echo "[Korali] Could not find an installation of ${libName}."
   exit 1
 fi

 echo "[Korali] Downloading ${libName}... "
 
 rm -rf $buildDir; check
 rm -rf $installDir; check
 
 mkdir -p $buildDir; check
 mkdir -p $installDir; check
 pushd $buildDir; check
 
 git clone https://github.com/SergioMartin86/libco.git $buildDir; check
  
 echo "[Korali] Building ${libName}... "
 make -j$NJOBS; check
 
 echo "[Korali] Installing ${libName}... "
 cp -r * $installDir; check;
 
 popd; check
 
 echo "[Korali] Finished installing ${libName}."
 binPath=${installDir}/bin/${binName}
 
 echo "[Korali] Cleaning up build folder..."
 rm -rf $buildDir; check
 
fi

######## Finalization ########

rm -f ${externalDir}/libcolink
ln -sf ${installDir} ${externalDir}/libcolink; check
echo "[Korali] Using libco located at ${installDir}."

exit 0
