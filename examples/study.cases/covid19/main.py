#!/usr/bin/env python3
# Author: George Arampatzis
# Date:   22/3/2020
# Email:  garampat@ethz.ch

import os
import argparse

from src.tools import tools
from src.model.sir.sample import *
from src.model.sir.propagate import *
from src.model.sir.intervals import *

parser = argparse.ArgumentParser()
aa = parser.add_argument
aa('--dataFolder',
   '-df',
   default='data',
   help='Save all results in this folder')
aa('--country',
   '-c',
   default='switzerland',
   help='Country from which to retrieve data./')
aa('--data', '-d', nargs='+', type=float, help='Infected population.')
aa('--populationSize',
   '-ps',
   type=int,
   default=80000,
   help='Total population.')
aa('--nSamples',
   '-ns',
   type=int,
   default=2000,
   help='Number of samples for TMCMC.')
aa('--nSamplesPropagation',
   type=int,
   default=10,
   help='Number of samples for propagation to compute intervals.')
aa('--nThreads', '-nt', type=int, default=1, help='Number of threads.')
aa('--nPoints',
   '-np',
   type=int,
   default=100,
   help='Number of points to evaluate the solution in the propagation phase.')
aa('--futureDays',
   '-fd',
   type=int,
   default=2,
   help=
   'Propagate that many days in future, after the time of observation of the last data.'
   )
aa('--validateData',
   '-vd',
   type=int,
   default=0,
   help=
   'Use that many data from the end of the data list to validate the prediction.'
   )
aa('--percentages',
   '-p',
   nargs='+',
   type=float,
   default=[0.5, 0.9],
   help='Percentages for confidence intervals.')
aa('--reduction',
   type=float,
   default=0.5,
   help='Reduction factor for R0 after intervention.')
aa('--duration',
   type=float,
   default=10,
   help='Duration of applying the  intervention.')
aa('--silent', action='store_true', help='No output on screen.')
aa('--moving_average',
   type=int,
   default=0,
   help='Half-width of moving average window applied to data.')
aa('--infer_reduction',
   action='store_true',
   help='Infer the reduction factor for R0 after intervention from the data.')
aa('--infer_duration',
   action='store_true',
   help='Infer the duration of intervention from the data.')
# aa('--noSave', action='store_true', help='No intermediate files saved.')
args = parser.parse_args()
tools.check_parsed_arguments(args)
args.noSave = False  # until we fix the bug in korali

args.dataFolder = os.path.join(os.path.abspath('.'), args.dataFolder) + '/'

# Download and save population data
jsData = tools.download_data(args)

# Sample the parameters of the computational model
jskSamples = sample_parameters(args, jsData)

# Propagate the uncertainty in the parameters
jskPropagation = propagate_uncertainty(args, jskSamples, jsData)

# Compute credible intervals
jskIntervals = compute_intervals(args, jskPropagation, jsData)
