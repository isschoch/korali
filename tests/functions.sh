###### Auxiliar Functions and Variables #########

curdir=$PWD
logFile=$curdir/test.log

# Function to check whether the last command was correctly executed
function check_result()
{
 if [ ! $? -eq 0 ]
 then 
  echo "[Korali] Error detected."
  exit -1
 fi 
}

# Logging and printing function.
function logEcho ()
{
 echo "$1"
 echo "$1" >> $logFile
}

# Logging function.
function log ()
{
 echo "$1" >> $logFile
}

