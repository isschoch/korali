Hierarchical Bayesian Inference (Basic)
=====================================================
In this tutorial we show how to perform hierarchical Bayesian inference.

Hierarchical Bayesian Inference is set up in 3 phases:
 - sample the posterior distributions conditioned on each data set
 - sample the hyper parameter
 - sample the posterior given hyperparameter and one (a) data set or  (b) data sets combined.

For each phase we provided an individual `python` file.

Running
---------------

All 3 phases can be run with the shell script `./run-hierarchical.sh`

The results are saved in sub dirs of the folder `/setup`.
