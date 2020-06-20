Integration
==================

In this tutorial we show how to **integrate** a given function. 

Problem Description
------------------- 

We are given the function :math:`f(x,y,z)=x^2+y^2+z^2` for :math:`x,y,z\in [0,1]^3`.
We want to find the integral of this function over its domain.

The Objective Function
----------------------

Create a folder named `model`. Inside, create a file with name `integrand.py` and paste the following code,

.. code-block:: python

        def integrand( sample ):
          x = sample["Parameters"][0] 
          #sample["Evaluation"] = x**2
          y = sample["Parameters"][1] 
          #sample["Evaluation"] = x**2+y**2
          z = sample["Parameters"][2]
          sample["Evaluation"] = x**2+y**2+z**2

This is the function we want to integrate.

Optimization with CMAES
-----------------------

First, open a file (you could name it 'run-integration.py') and import the korali module

.. code-block:: python

    #!/usr/bin/env python3
    import korali

Import the integrand,

.. code-block:: python

    import sys
    sys.path.append('./model')
    from integrand import *

The Korali Engine and Experiment Objects
----------------------------------------

Next we construct a `korali.Engine` and a `korali.Experiment` object and set the computational model,

.. code-block:: python

    k = korali.Engine()
    e = korali.Experiment()

The Problem Type
----------------

Then, we set the type of the problem to `Integration`, set the function to integrate and chose the integration method

.. code-block:: python

        e["Problem"]["Type"] = "Integration"
        e["Problem"]["Integrand"] = lambda modelData: integrand(modelData)
        e["Problem"]["Integration Method"] = "Rectangle"#"Rectangle" #"Trapezoidal", "Simpson", "Monte Carlo", "Custom"

The Variables
-------------

In this problem there is three variables, `X`, `Y` and `Z`, whose domain we set to [0,1] and in case of Monte Carlo Integration assume an uniform distribution. Furthermore we assume 10 samples per dimension

.. code-block:: python

        e["Variables"][0]["Name"] = "x"
        e["Variables"][0]["Number Of Gridpoints"] = 10
        e["Variables"][0]["Lower Bound"] = 0.0
        e["Variables"][0]["Upper Bound"] = 1.0
        e["Variables"][0]["Sampling Distribution"] = "Uniform" 
        
        e["Variables"][1]["Name"] = "y"
        e["Variables"][1]["Lower Bound"] = 0.0
        e["Variables"][1]["Upper Bound"] = 1.0
        e["Variables"][1]["Number Of Gridpoints"] = 10
        e["Variables"][1]["Sampling Distribution"] = "Uniform" 
        
        e["Variables"][2]["Name"] = "z"
        e["Variables"][2]["Lower Bound"] = 0.0
        e["Variables"][2]["Upper Bound"] = 1.0
        e["Variables"][2]["Number Of Gridpoints"] = 10
        e["Variables"][2]["Sampling Distribution"] = "Uniform" 

The Solver
----------
We choose the solver `Integrator`, don't set the execution per generation, to have the summation be performed in one generation,

.. code-block:: python

        e["Solver"]["Type"] = "Integrator"
        #e["Solver"]["Executions Per Generation"] = 100

For a detailed description of Integrator settings see :ref:`Integrator <module-solver-integrator>`.

Finally, we need to add a call to the run() routine to start the Korali engine.

.. code-block:: python

    k.run(e)

Running
-------

We are now ready to run our example:
`./run-integration`

Or, alternatively:
`python3 ./run-integration.py`

The results are saved in the folder `_korali_result/`.
