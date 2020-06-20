#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil


def runLAMMPS(x):
	
	SourceFolderName = "setup/model"
	DestinationFolderName = os.environ['SCRATCH'] + '/lammps' + str(os.environ['SLURM_JOBID']) + '/' + 'result' + str(x["Sample Id"])
	ResultsFile = "results.txt"

	
	# Copy the 'model' folder into a temporary directory
	if os.path.exists( DestinationFolderName ):
		shutil.rmtree( DestinationFolderName)
	shutil.copytree( SourceFolderName, DestinationFolderName )

	CurrentDirectory = os.getcwd()

	# Move inside the temporary directory
	try:
		os.chdir( DestinationFolderName )
	except OSError as e:
		print("I/O error(" + str(e.errno) + "): " + e.strerror )
		print("The folder " + DestinationFolderName + " is missing")
		sys.exit(1)
	
	# Write the parameters from Korali into the file 'params.txt'.
	# LAMMPS will get the parameters from this file.
	f = open("params.txt","w")
	for i in range(2):
		f.write("%.10f\n" % x["Parameters"][i] )
	f.close()

	# Run the script that calls LAMMPS and post-processes the resuls
	subprocess.call("./doall.sh")

	# Read the Loglikelihood value
	Y = readColumnFromFile( ResultsFile, 0 )
 
	sig = x["Parameters"][2]

	result = []
	sdev   = []
	for y in Y: 
                result.append(float(y))
                sdev.append(sig*y)

	x["Reference Evaluations"] = result
	x["Standard Deviation"]    = sdev

	# Move back to the base directory
	os.chdir( CurrentDirectory )

	# Delete the temporary directory
	if os.path.exists( DestinationFolderName ):
		shutil.rmtree( DestinationFolderName)


def getReferenceData():
	y = readColumnFromFile( "setup/data/data.txt", 1 )
	return y


def readColumnFromFile( FileName, Column ): 
	try:
		f = open( FileName, "r" )
		lines = f.readlines()
		y = []
		for l in lines:
			y.append(float(l.split()[Column]))
		f.close()
	except IOError as e:
		print("I/O error(" + str(e.errno) + "): " + e.strerror )
		print("The file " + FileName + " is missing")
		sys.exit(1)
	except ValueError:
		print( "Could not convert data to a float. Check the results in " + FileName )
		sys.exit(1)
	except:
		print( "Unexpected error: " + sys.exc_info()[0] )
		raise
		
	return y
