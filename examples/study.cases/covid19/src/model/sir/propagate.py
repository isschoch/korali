#!/usr/bin/env python3
# Author: George Arampatzis
# Date:   22/3/2020
# Email:  garampat@ethz.ch
# Description: Run the SIR model on all the parameters obtained from the sampling.
#              Save the solution of each model in a separate file in the folder
#              '_executor_output/'
import json
import shutil
import numpy as np
import korali

from . import sir
from ...tools import tools


def propagate_uncertainty(args, jskSamples, jsData):
    sir.set_custom_params(reduction=args.reduction,
                          infer_reduction=args.infer_reduction,
                          duration=args.duration,
                          infer_duration=args.infer_duration)
    jsOde = sir.getReferenceData(jsData)

    Ns = jskSamples['Solver']['Population Size']
    Np = len(jskSamples['Samples'][0]['Parameters'])

    db = jskSamples['Results']['Sample Database']
    p = []
    for j in range(Np):
        tmp = []
        for k in range(Ns):
            tmp.append(db[k][j])
        p.append(tmp)

    # days of prediction = days in the data + future days
    T = jsOde['Time'][-1] + args.futureDays
    t = np.linspace(0, T, args.nPoints)
    jsOde['Time'] = t.tolist()

    e = korali.Experiment()

    e['Problem']['Type'] = 'Propagation'
    e['Problem']['Execution Model'] = \
                lambda modelData: sir.model_for_korali_execute(
            modelData, jsOde)

    for i, var in enumerate(sir.params_to_infer):
        e['Variables'][i]['Name'] = var
        e['Variables'][i]['Precomputed Values'] = p[i]

    e['Solver']['Type'] = 'Executor'

    e['File Output'] = tools.fileOutputDefaults(args, '_korali_propagation/')

    if (args.silent):
        e['Console Output']['Verbosity'] = 'Silent'

    e['Store Sample Information'] = True

    k = korali.Engine()

    k['Conduit']['Type'] = 'Concurrent'
    k['Conduit']['Concurrent Jobs'] = args.nThreads

    k.run(e)

    return e
