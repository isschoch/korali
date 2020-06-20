% Add the following to your startup.m:
% addpath(genpath('KORALI_PATH/source/matlab/'))

% file = 'results_phase_1/004/final.json';
% file = 'results_phase_2/final.json';
file = 'results_phase_3a/final.json';
% file = 'results_phase_3b/final.json';

data = jsondecode(fileread(file));

N = length(data.Variables); 

Ns = data.Solver.PopulationSize; 

x = data.Solver.Internal.SampleDatabase;
y = data.Solver.Internal.SampleLoglikelihoodDatabase;

figure(1); clf
[ h, ax, BigAx, hhist, pax ] = plotmatrix_hist( x , y );
