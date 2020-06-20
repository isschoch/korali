#!/usr/bin/env python3
# Author: George Arampatzis
# Date:   16/3/2020
# Email:  garampat@ethz.ch
# Description: Compute and plot credible intervals

import json
import numpy as np
import argparse

import matplotlib.pyplot as plt
from src.tools.tools import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataFolder',
                    '-df',
                    default='data',
                    help='Load results from this folder')
args = parser.parse_args()

with open(args.dataFolder + '/intervals.json') as f:
    d = json.load(f)

var = ['Total Infected', 'I', 'R']

y_label = ['Total Infected', 'New Infected', 'Total Recovered']

fig = plt.figure(figsize=(12, 8))
ax = fig.subplots(len(var))

ax[0].plot(d['x-data'],
           d['y-data'],
           'o',
           lw=2,
           label='infected (data)',
           color='black')

ax[0].plot(d['x-data Validate'],
           d['y-data Validate'],
           'x',
           lw=2,
           label='validation data',
           color='black')

for k, y in enumerate(var):
    ax[k].plot(d['x-axis'],
               d[y]['Mean'],
               '-',
               lw=2,
               label='Mean',
               color='blue')
    ax[k].plot(d['x-axis'],
               d[y]['Median'],
               '-',
               lw=2,
               label='Median',
               color='black')

    d[y]['Intervals'].sort(key=lambda x: x['Percentage'], reverse=True)

    for x in d[y]['Intervals']:
        p1 = [max(k, 0) for k in x['Low Interval']]
        p2 = x['High Interval']
        p = 100 * x['Percentage']
        ax[k].fill_between(d['x-axis'],
                           p1,
                           p2,
                           alpha=0.5,
                           label=f' {p:.0f}% credible interval')

    ax[k].legend(loc='upper left')
    ax[k].set_ylabel(y_label[k])
    ax[k].set_xticks(range(np.ceil(max(d['x-axis']) + 1).astype(int)))
    ax[k].grid()

ax[-1].set_xlabel('time in days')

res_folder = args.dataFolder + '/figures/'
prepareFolder(res_folder, clean=False)
fig.savefig(res_folder + 'prediction.png')

_, samples_file = get_last_generation(args.dataFolder + "/_korali_samples/",
                                      'gen*.json')
with open(samples_file) as json_file:
    data = json.load(json_file)
p = data["Results"]["Sample Database"]
beta = [row[0] for row in p]
gamma = [row[1] for row in p]

fig = plt.figure(figsize=(12, 8))
ax = fig.subplots(1)

R0 = np.asarray(beta) / np.asarray(
    gamma) * d['S']['Mean'][0] / d['Population Size']

ax.hist(R0, 100, density=True, facecolor='g', alpha=0.75)
ax.set_xlabel('R0')
ax.grid()

fig.savefig(res_folder + 'R0.png')

plt.show()
