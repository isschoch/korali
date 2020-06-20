Model Propagation
==========================================

In this tutorial we show how to pass multiple input sets through a computational model (this setup may be used to do uncertainty propagation and distribute the workload).

Problem Description
---------------------------

We are given a set of parameters to evaluate in a file 'samplesOut.dat'.
We want to execute a model function :math:`f(\theta)` on given parameters.

Propagate Model Evaluations
---------------------------

The Korali Object
---------------------------
Initialize a korali object

.. code-block:: python

    e = korali.Experiment()
 
The Problem Type
---------------------------
The type of problem is `Execution/Model`.

.. code-block:: python

    e["Problem"]["Type"] = "Execution/Model"

The Variables
---------------------------

In the file we have means and variances to evaluate:

.. code-block:: python

    e["Variables"][0]["Name"] = "Mean"
    e["Variables"][0]["Loaded Values"] = means
    e["Variables"][1]["Name"] = "Variance"
    e["Variables"][1]["Loaded Values"] = variances

The Solver
---------------------------
We set the solver and choose how many samples are evaluated per generation.

.. code-block:: python

    e["Solver"]["Type"] = "Executor"
    e["Solver"]["Executions Per Generation"] = 1

Running
---------------------------
We are now ready to run our example: `./run-execution.py`

The results are saved in the folder `_korali_result/`.
