#!/usr/bin/env python3
# Author: George Arampatzis
# Date:   22/3/2020
# Email:  garampat@ethz.ch

import korali

from . import sir
from ...tools import tools


def sample_parameters(args, jsData):
    sir.set_custom_params(reduction=args.reduction,
                          infer_reduction=args.infer_reduction,
                          duration=args.duration,
                          infer_duration=args.infer_duration)
    jsOde = sir.getReferenceData(jsData)

    # Creating new experiment
    e = korali.Experiment()

    # Setting up the reference likelihood for the Bayesian Problem
    e['Problem']['Type'] = 'Bayesian/Reference'
    e['Problem']['Likelihood Model'] = sir.likelihood
    e['Problem']['Reference Data'] = jsOde['Data']
    e['Problem'][
        'Computational Model'] = lambda sampleData: sir.model_for_korali_sample(
            sampleData, jsOde)

    # Configuring TMCMC parameters
    e['Solver']['Type'] = 'TMCMC'
    e['Solver']['Population Size'] = args.nSamples

    for i, var in enumerate(sir.params_to_infer):
        distname = 'Uniform ' + var
        e['Distributions'][i]['Name'] = distname
        e['Distributions'][i]['Type'] = 'Univariate/Uniform'
        minmax = sir.params_prior[var]
        e['Distributions'][i]['Minimum'] = minmax[0]
        e['Distributions'][i]['Maximum'] = minmax[1]
        e['Variables'][i]['Name'] = var
        e['Variables'][i]['Prior Distribution'] = distname

    if (args.silent):
        e['Console Output']['Verbosity'] = 'Silent'
    else:
        e['Console Output']['Verbosity'] = 'Detailed'

    e['File Output'] = tools.fileOutputDefaults(args, '_korali_samples/')
    e['File Output']['Frequency'] = 50
    e["Store Sample Information"] = True

    k = korali.Engine()

    k['Conduit']['Type'] = 'Concurrent'
    k['Conduit']['Concurrent Jobs'] = args.nThreads

    k.run(e)

    return e
