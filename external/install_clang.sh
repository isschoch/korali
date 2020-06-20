#!/bin/bash

######### Global Definitions ########
libName="llvm"
binName="clang-format"
minVersion=10.0.0

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
baseLibDir=${externalDir}/_${libName}
installDir=${baseLibDir}/install
buildDir=${baseLibDir}/build

binFound=0
binPath=${installDir}/bin/${binName}

if [ ! -f ${binPath} ]; then
 binPath=${binName}
fi

$binPath --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
 binVersion=`${binPath} --version | head -n 1`
 cmpver=`printf "${binVersion}\n${minVersion}" | sort -V | head -n 1 | cut -d' ' -f 3`
 
 if [[ "$cmpver" != "$minVersion" ]]; then
    echo "[Korali] ${libName} version found (${binVersion}) is smaller than required (${minVersion})."
 else
    binFound=1
    echo "[Korali] Found existing ${libName} version ${binVersion}. Skipping installation..."
 fi
fi

######## If not installed, download and install ########

if [ ${binFound} == 0 ]; then

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
  
 arch="$(uname -s)"; check
 
 # If using MacOs, use the Darwin package 
 if [ "$arch" == "Darwin" ]; then
  
  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-apple-darwin.tar.xz;  check
  
  tar -xf clang+llvm-10.0.0-x86_64-apple-darwin.tar.xz;  check
  
  mv clang+llvm-10.0.0-x86_64-apple-darwin/* $installDir; check
 
 else  # Else default to Linux64

  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-sles11.3.tar.xz; check
  
  tar -xf clang+llvm-10.0.0-x86_64-linux-sles11.3.tar.xz;  check
  
  mv clang+llvm-10.0.0-x86_64-linux-sles11.3 llvm;  check
  
  # Fix for systems that have no libcurses lib
  ln -s libunwind.so.1.0 llvm/lib/libncurses.so.5;  check
  
  # Finally copy it to destination
  mv llvm/* $installDir;  check
  
 fi
 
 popd; check
 
 echo "[Korali] Finished installing ${libName}."
 binPath=${installDir}/bin/${binName}
 
 echo "[Korali] Cleaning up build folder..."
 rm -rf $buildDir; check
fi

######## Finalization ######## 

fullBinPath=`which ${binPath}`
ln -sf $fullBinPath ${externalDir}/${binName}; check
versionLine=`${externalDir}/${binName} --version`; check
binVersion=`echo $versionLine | head -n 1 | cut -d' ' -f 3`; check 
echo "[Korali] Using ${libName} version $binVersion"

exit 0
