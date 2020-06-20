Discrete Optimization: Searching the Global Maximum
===================================================

In this tutorial we show how to optimize a function with discrete input parameters along with continuous ones.

Problem Description
---------------------------

We want to minimize the following function:

.. math:

   f(x_1, ... x_{10}) = \sum_{i=1}^{10} 10^{6*i/10} * x_i^2 $$ ,
   where $x_1, x_2, x_4, x_7 \in \mathbb{Z}$, $x_3, x_5, x_6, x_9, x_{10} \in \mathbb{R}$.

The Objective Function
---------------------------
We create a folder `model`, and inside a file `model.py`, where we define the
function that we want to optimize:

.. code-block:: python

    def model(d):
      npar = 10
      res = 0.0
      v = d["Parameters"]

      for i in range(npar):
        if( i == 0 or i == 1 or i == 3 or i == 6):
            res += pow( 10, 6.0*i/npar) * round(v[i]) * round(v[i])
        else:
            res += pow( 10, 6.0*i/npar) * v[i] * v[i]

      d["Evaluation"] = -res;

Then, in another file, for example `run-cmaes.py`, we start by importing the
function we just defined (assuming `model.py` is in subfolder `model` relative
to the current file), and creating an `Experiment` which we will configure,

.. code-block:: python

    import sys
    sys.path.append('model')
    from model import *

    import korali
    e = korali.Experiment()

The Problem Type
---------------------------
We choose *direct evaluation* as problem type, set the objective function to
our previously defined `model()`, and choose maximization as objective,

.. code-block:: python

    e["Problem"]["Type"] = "Evaluation/Direct/Basic"
    e["Problem"]["Objective"] = "Maximize"
    e["Problem"]["Objective Function"] = model

The Variables
---------------------------
We define 10 variables, of which four (:math:`x_1, x_2, x_4, x_7`) are discrete. Also,
we limit their domain to [-19, 21] each.

.. code-block:: python

    for i in range(10) :
      e["Variables"][i]["Name"] = "X" + str(i)
      e["Variables"][i]["Initial Mean"] = 1.0
      e["Variables"][i]["Lower Bound"]  = -19.0
      e["Variables"][i]["Upper Bound"]  = +21.0

    # We set some of them as discrete.
    e["Variables"][0]["Granularity"] = 1.0
    e["Variables"][1]["Granularity"] = 1.0
    e["Variables"][3]["Granularity"] = 1.0
    e["Variables"][6]["Granularity"] = 1.0

The Solver
---------------------------
We choose the solver `CMA-ES` and set two termination criteria,

.. code-block:: python

    e["Solver"]["Type"] = "Optimizer/CMAES"
    e["Solver"]["Population Size"] = 8
    e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-9
    e["Solver"]["Termination Criteria"]["Max Generations"] = 5000

Output configuration
---------------------------
To redcue output frequency of result files and on the console we set

.. code-block:: python

    e["File Output"]["Frequency"] = 50
    e["Console Output"]["Frequency"] = 50

The Korali Engine Object
---------------------------
We create a Korali engine, and tell it to run the experiment we defined,

.. code-block:: python

    k = korali.Engine()
    k.run(e)


Running
---------------------------
We are now ready to run our example: `./run-cmaes.py`


The results are saved in the folder `_korali_result/`.

Plotting
---------------------------

You can see the results of CMA-ES by running the command, `python3 -m korali.plotter`
which visualizes the results found in folder `_korali_result`.
