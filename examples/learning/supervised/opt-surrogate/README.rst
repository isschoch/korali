Optimization on Surrogates with Gaussian Processes
==================================================

This tutorial shows how to use a korali surrogate as a model for optimization.
The file ``train-surrogate.py`` constructs a Gaussian Process surrogate fronm the given synthetic data and saves it inside the ``_korali_result_surrogate/`` folder.
Second, we load and use the trained surrogate as a function to optimize.
This is done in ``run-cmaes.py``.
