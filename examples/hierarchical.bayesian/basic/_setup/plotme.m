% Add the following to your startup.m:
% addpath(genpath('KORALI_PATH/source/matlab/'))

% file = 'results_phase_1/001/final.json';
% file = 'results_phase_2/final.json';
% file = 'results_phase_3a/final.json';
file = 'results_phase_3b/final.json';

data = jsondecode(fileread(file));

N = length(data.Problem.Variables); 

Ns = data.Solver.PopulationSize; 

x = reshape( data.Solver.Internal.SampleDatabase, N, Ns )';
y = data.Solver.Internal.SampleFitnessDatabase;

figure(1); clf
[ h, ax, BigAx, hhist, pax ] = plotmatrix_hist( x , y );