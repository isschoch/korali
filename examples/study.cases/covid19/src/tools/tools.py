#!/usr/bin/env python3
# Author: George Arampatzis
# Date:   16/3/2020
# Email:  garampat@ethz.ch

import os
import sys
import shutil
import glob

import pandas as pd
import io
import requests
import json
import os.path
import numpy as np


def print_params_to_infer(model, pairs=False):
    """Print the list of parameter identifiers for 'model' to stdout
    """
    try:
        tags = model.params_to_infer
    except AttributeError:
        raise RuntimeError(f"Model {model} has no 'params_to_infer' attribute")
    try:
        pretty = model.params_to_infer_pretty
    except AttributeError:
        pretty = tags
    if pairs:
        print(' '.join([f'{x},{y}' for x,y in zip(tags,pretty)]))
    else:
        print(' '.join([x for x in tags]))



def prepareFolder( dir, clean=True):
  if(clean==True):
    shutil.rmtree(dir, ignore_errors=True)
  if os.path.isdir(dir) == False:
    os.mkdir(dir)




def get_last_generation( folder, pattern ):
  files = glob.glob(folder + pattern)
  return len(files), sorted(files)[-1]




def check_parsed_arguments( args ):
  if(args.data and len(args.data)==0 ):
    sys.exit('Empty data list.')
  if( args.nThreads<1 ):
    sys.exit('Number of threads must be >= 1.')
  if( args.nSamples < 10 ):
    sys.exit('Number of samples must >= 10.')
  if( args.populationSize < 2):
    sys.exit('Population size must be > 1')
  if( args.nPoints < 1 ):
    sys.exit('Number of samples must >= 1.')
  if( args.futureDays < 0 ):
    sys.exit('Number of samples must > 0.')
  if(  any( [ x<=0. or x>=1. for x in args.percentages ] ) ):
    sys.exit('Percentages must be between 0 and 1.')
  args.percentages.sort()




def fileOutputDefaults( args, saveFolder):
  js = {}
  if(args.noSave):
    js['Enabled'] = False
  else:
    js['Enabled'] = True
    saveFolder = os.path.join(args.dataFolder, saveFolder)
    prepareFolder( saveFolder )
    relativeSaveFolder = os.path.relpath(saveFolder, './')
    js['Path'] = relativeSaveFolder

  return js


def moving_average(x, w):
    """
    x: `numpy.ndarray`, (N)
    w: int
      Window half-width.
    Returns:
    xa: `numpy.ndarray`, (N)
      Array `x` averaged over window [-w,w].
    """
    s = np.zeros_like(x)
    q = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(max(0, i - w), min(i + w + 1, len(x))):
            if np.isfinite(x[j]):
                s[i] += x[j]
                q[i] += 1
    xa = s / q
    return xa




def download_data( args ):
  if( args.data ):
    I = args.data
  else:
    url = 'https://hgis.uw.edu/virus/assets/virus.csv'
    print(f'Retrieve population data for {args.country} from: {url}')

    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    if( not args.country in list( df.columns.values ) ):
      sys.exit('Country not in database.')
    d = df[[args.country]].dropna().values.tolist()
    I = [ float(l.split('-')[0]) for  k in d for l in k ]

  Nv = args.validateData
  N  = len(I)-Nv
  I = list(moving_average(I, args.moving_average))
  Iv = I[-Nv:] if Nv>0 else []
  I  = I[:N]

  js={}
  js['Population Size'] = args.populationSize
  js['Time'] = [ i for i in range(N) ]
  js['Infected'] = I
  js['Infected Validation'] = Iv
  js['Time Validation'] = [ N+i for i in range(Nv) ]
  js['Country'] = args.country

  if( not args.noSave ):
    prepareFolder( args.dataFolder )
    dataFile = args.dataFolder + 'population_data.json'
    print(f'Save population data in: {dataFile}')
    with open(dataFile, 'w') as f:
      json.dump( js, f, indent=2, sort_keys=True)

  return js
