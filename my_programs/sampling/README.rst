Sampling
======================================

In this tutorial we show how to directly **sample** from a function
using Metropolis Hastings (MCMC).

Problem Description
---------------------------

We are given the function :math:`g(\vartheta)=\exp(-\vartheta^2)` for :math:`\vartheta\in[-10,10]`.

We assume that :math:`f` represents the *unnormalized* density of a distribution.
We want to draw samples from this distribution.

For the rest of the tutorial we will work with the function :math:`f(\vartheta)=\log g(\vartheta) = -\vartheta^2` for numerical reasons.
In general we advise users of Korali to work in log space.

The Objective Function
---------------------------

Create a folder named `model`. Inside, create a file with name `directModel.py` and paste the following code,

.. math::

    #!/usr/bin/env python

    def evaluateModel( x ):
      v = x["Parameters"][0]
      x["Evaluation"] = -v*v


This is the computational model that represents our objective function.

Sampling with MCMC
---------------------------

First, open a file and import the korali module

.. code-block:: python

    #!/usr/bin/env python3
    import korali

Import the computational model,

.. code-block:: python

    import sys
    sys.path.append('./model')
    from directModel import *


The Korali Experiment Object
----------------------------------

Next we construct a `korali.Experiment` object and set the computational model,

.. code-block:: python

    e = korali.Experiment()
    e["Problem"]["Objective Function"] = model

The Problem Type
---------------------------
Then, we set the type of the problem to `Direct Evaluation`

.. code-block:: python

    e["Problem"]["Type"] = "Evaluation/Direct/Basic"

The Variables
---------------------------
In this problem there is only one variable,

.. code-block:: python

    e["Variables"][0]["Name"] = "X"

The Solver
---------------------------
We choose the solver `MCMC` and set the initial mean and standard deviation of the parameter `X`.

.. code-block:: python

    e["Solver"]["Type"]  = "MCMC"
    e["Variables"][0]["Initial Mean"] = 0.0
    e["Variables"][0]["Initial Standard Deviation"] = 1.0


    e["Solver"]["Burn In"] = 500
    e["Solver"]["Termination Criteria"]["Max Samples"] = 5000

We also set some settings for MCMC. For a detailed description of the MCMC settings, see
:ref:`MCMC <module-solver-sampler-mcmc>`

Configuring the output
---------------------------
To reduce the output frequency we write

.. code-block:: python

    e["File Output"]["Frequency"]    = 500
    e["Console Output"]["Frequency"] = 500
    e["Console Output"]["Verbosity"] = "Detailed"

Running
---------------------------

Finally, we are ready to run the simulation,

.. code-block:: python

    k = korali.Engine()
    k.run(e)

The results are saved in the folder `_korali_result/`.


Plotting
---------------------------
You can see a histogram of the results by running the command `python3 -m korali.plotter`

