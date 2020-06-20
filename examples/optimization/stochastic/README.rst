Optimization: Searching the Global Maximum
==========================================

In this tutorial we show how to **optimize** a given function. 

Problem Description
------------------- 

We are given the function :math:`f(\vartheta)=-\vartheta^2` for :math:`\vartheta\in[-10,10]`.
We want to find the maximum of the function in the given interval.

The Objective Function
----------------------

Create a folder named `model`. Inside, create a file with name `directModel.py` and paste the following code,

.. code-block:: python

    #!/usr/bin/env python

    def evaluateModel(p):
        x = p["Parameters"][0]
        p["Evaluation"] = -x*x

This is the computational model that represents our objective function.


Optimization with CMAES
-----------------------

First, open a file (you could name it 'run-cmaes.py') and import the korali module

.. code-block:: python

    #!/usr/bin/env python3
    import korali

Import the computational model,

.. code-block:: python

    import sys
    sys.path.append('./model')
    from directModel import *

The Korali Engine and Experiment Objects
----------------------------------------

Next we construct a `korali.Engine` and a `korali.Experiment` object and set the computational model,

.. code-block:: python

    k = korali.Engine()
    e = korali.Experiment()

    e["Problem"]["Objective Function"] = evaluateModel


The Problem Type
----------------

Then, we set the type of the problem to `Direct Evaluation`, and the objective to maximization,

.. code-block:: python

    e["Problem"]["Type"] = "Evaluation/Direct/Basic"
    e["Problem"]["Objective"] = "Maximize"

The Variables
-------------

In this problem there is only one variable, `X`, whose domain we set to [-10,10],

.. code-block:: python

    e["Variables"][0]["Name"] = "X"
    e["Variables"][0]["Lower Bound"] = -10.0
    e["Variables"][0]["Upper Bound"] = +10.0

The Solver
----------
We choose the solver `CMAES`, set the population size to be `32` and two termination criteria,

.. code-block:: python

    e["Solver"]["Type"] = "Optimizer/CMAES"
    e["Solver"]["Population Size"] = 32
    e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7
    e["Solver"]["Termination Criteria"]["Max Generations"] = 100

For a detailed description of CMAES settings see :ref:`CMAES <module-solver-optimizer-cmaes>`.

Finally, we need to add a call to the run() routine to start the Korali engine.

.. code-block:: python

    k.run(e)

Running
-------

We are now ready to run our example:
`./run-cmaes`

Or, alternatively:
`python3 ./run-cmaes`

The results are saved in the folder `_korali_result/`.

Plotting
--------
You can see the results of CMA-ES by running the command,
`python3 -m korali.plotter`
