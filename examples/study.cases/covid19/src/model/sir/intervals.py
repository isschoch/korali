#!/usr/bin/env python3
# Author: George Arampatzis
# Date:   16/3/2020
# Email:  garampat@ethz.ch
# Description: Compute and plot credible intervals

import json
import numpy as np
from scipy.stats import norm
from scipy import optimize
import argparse
import json

from . import sir
from .sir import Likelihood
from ...tools import tools


def compute_credible_intervals(ydata,
                               std,
                               like: Likelihood,
                               percentages,
                               ns=10,
                               cumsum=False):
    ydata = np.asarray(ydata)
    std = np.asarray(std)

    Ns = ydata.shape[0]
    Nt = ydata.shape[1]

    samples = np.zeros((Ns * ns, Nt))

    iv = {}
    iv['Intervals'] = []
    if like == Likelihood.normal:
        for k in range(Nt):
            m = ydata[:, k]
            r = std[:, k]
            x = [np.random.normal(m, r) for _ in range(ns)]
            samples[:, k] = np.asarray(x).flatten()
    elif like == Likelihood.nbin:
        for k in range(Nt):
            m = ydata[:, k]
            r = std[:, k]
            p = np.clip(m / (m + r), 0, 1)
            x = [np.random.negative_binomial(r, 1 - p) for _ in range(ns)]
            samples[:, k] = np.asarray(x).flatten()
    else:
        raise NotImplementedError()

    if cumsum:
        samples = np.cumsum(samples, axis=1)

    median = np.zeros(Nt)
    mean = np.zeros(Nt)
    for k in range(Nt):
        median[k] = np.quantile(samples[:, k], 0.5)
        mean[k] = np.mean(samples[:, k])

    iv['Median'] = list(median)
    iv['Mean'] = list(mean)

    for p in percentages:
        q1 = np.zeros((Nt, ))
        q2 = np.zeros((Nt, ))
        for k in range(Nt):
            q1[k] = np.quantile(samples[:, k], 0.5 - p / 2)
            q2[k] = np.quantile(samples[:, k], 0.5 + p / 2)

        local_iv = {}
        local_iv['Percentage'] = p
        local_iv['Low Interval'] = list(q1)
        local_iv['High Interval'] = list(q2)
        iv['Intervals'].append(local_iv)

    return iv


def intervals_cumsum(iv_daily, exp=2.):
    """
    Returns intervals computed as cumulative sum of iv.
    """
    iv = {}
    iv['Intervals'] = []
    iv['Median'] = list(np.nancumsum(iv_daily['Median']))
    iv['Mean'] = list(np.nancumsum(iv_daily['Mean']))
    for local_iv_daily in iv_daily['Intervals']:
        local_iv = {}
        p = local_iv_daily['Percentage']
        local_iv['Percentage'] = p
        local_iv['Low Interval'] = []
        local_iv['High Interval'] = []

        def cumsum_iv(iv, mean):
            iv = np.array(iv)
            mean = np.array(mean)
            return (np.nancumsum(abs(mean - iv)**exp))**(1. / exp)

        def cumsum_low(low, mean, meancum):
            meancum = np.array(meancum)
            return list(meancum - cumsum_iv(low, mean))

        def cumsum_high(high, mean, meancum):
            meancum = np.array(meancum)
            return list(meancum + cumsum_iv(high, mean))

        local_iv['Low Interval'] = cumsum_low(local_iv_daily['Low Interval'],
                                              iv_daily['Mean'], iv['Mean'])
        local_iv['High Interval'] = cumsum_high(
            local_iv_daily['High Interval'], iv_daily['Mean'], iv['Mean'])

        iv['Intervals'].append(local_iv)
    return iv


def intervals_diff(ivcum):
    """
    Returns intervals computed as diff of iv.
    """
    def diff(x):
        return np.diff(x, prepend=0)

    iv = {}
    iv['Intervals'] = []
    iv['Median'] = list(diff(ivcum['Median']))
    iv['Mean'] = list(diff(ivcum['Mean']))
    for local_ivcum in ivcum['Intervals']:
        local_iv = {}
        p = local_ivcum['Percentage']
        local_iv['Percentage'] = p
        local_iv['Low Interval'] = []
        local_iv['High Interval'] = []

        def diff_iv(ivcum, meancum):
            ivcum = np.array(ivcum)
            meancum = np.array(meancum)
            return ((np.mean(abs(ivcum - meancum)))**2 / len(ivcum))**0.5

        def diff_low(lowcum, meancum, mean):
            return list(mean - diff_iv(lowcum, meancum))

        def diff_high(highcum, meancum, mean):
            return list(mean + diff_iv(highcum, meancum))

        local_iv['Low Interval'] = diff_low(local_ivcum['Low Interval'],
                                            ivcum['Mean'], iv['Mean'])
        local_iv['High Interval'] = diff_high(local_ivcum['High Interval'],
                                              ivcum['Mean'], iv['Mean'])
        iv['Intervals'].append(local_iv)
    return iv


def get_mean_parameters(args):
    from glob import glob
    import json
    import os
    gens = glob(os.path.join(args.dataFolder, "_korali_samples", "gen*.json"))
    lastgen = sorted(gens)[-1]
    with open(lastgen) as f:
        js = json.load(f)
    return js['Solver']['Mean Theta']


def compute_intervals(args, jskPropagation, jsData):
    sir.set_custom_params(reduction=args.reduction,
                          infer_reduction=args.infer_reduction,
                          duration=args.duration,
                          infer_duration=args.infer_duration)
    jsOde = sir.getReferenceData(jsData)

    r = args.percentages
    Ns = args.nSamples
    Nsp = args.nSamplesPropagation

    if (args.silent == False): print('Loading files...')

    Nv = jskPropagation['Samples'][0]['Saved Results']['Number of Variables']
    Nt = jskPropagation['Samples'][0]['Saved Results']['Length of Variables']

    varNames = []
    derived = []
    for k in range(Nv):
        varNames.append(jskPropagation['Samples'][0]['Saved Results']
                        ['Variables'][k]['Name'])
        derived.append(jskPropagation['Samples'][0]['Saved Results']
                       ['Variables'][k]['derived'])

    # FIXME: too slow
    vars = {}
    for i, x in enumerate(varNames):
        vars[x] = []
        for k in range(Ns):
            vars[x].append(jskPropagation['Samples'][k]['Saved Results']
                           ['Variables'][i]['Values'])

    t = jskPropagation['Samples'][0]['Saved Results']['x-axis']

    if (args.silent == False):
        print('Finished loading files.')
        print('Calculate credible intervals...')

    vars_ci = {}
    if sir.likelihood == sir.Likelihood.nbin:
        std = [
            jskPropagation['Samples'][k]['Saved Results']['Dispersion']
            for k in range(Ns)
        ]
    elif sir.likelihood == sir.Likelihood.normal:
        std = [
            jskPropagation['Samples'][k]['Saved Results']['Standard Deviation']
            for k in range(Ns)
        ]
    else:
        raise NotImplementedError()
    for x in varNames:
        vars_ci[x] = compute_credible_intervals(vars[x],
                                                std,
                                                sir.likelihood,
                                                r,
                                                ns=Nsp)

    for x, d in zip(varNames, derived):
        if d:
            if d[0] == 'cumsum':
                vars_ci[x] = compute_credible_intervals(vars[d[1]],
                                                        std,
                                                        sir.likelihood,
                                                        r,
                                                        ns=Nsp,
                                                        cumsum=True)
            elif d[0] == 'diff':
                vars_ci[x] = intervals_diff(vars_ci[d[1]])
            else:
                raise NotImplementedError()

    if (args.silent == False):
        print('Finished calculating credible intervals...')

    # make json and write it to file
    jsIntervals = vars_ci
    jsIntervals['x-axis'] = t
    jsIntervals['x-data'] = jsData['Time']
    jsIntervals['y-data'] = jsData['Infected']
    jsIntervals['x-data Validate'] = jsData['Time Validation']
    jsIntervals['y-data Validate'] = jsData['Infected Validation']
    jsIntervals['Population Size'] = args.populationSize
    jsIntervals['nSamples'] = args.nSamples
    jsIntervals['percentages'] = args.percentages
    jsIntervals['reduction'] = args.reduction
    jsIntervals['mean_params'] = \
            sir.params_list_to_dict(get_mean_parameters(args))

    if (not args.noSave):
        tools.prepareFolder(args.dataFolder, clean=False)
        dataFile = args.dataFolder + 'intervals.json'
        print(f'Save intervals in: {dataFile}')

        with open(dataFile, 'w') as f:
            json.dump(jsIntervals, f, indent=2, sort_keys=True)

    return jsIntervals
