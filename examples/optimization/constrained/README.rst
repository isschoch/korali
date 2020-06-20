Constrained Optimization: Searching the Global Maximum
======================================================

In this tutorial we show how to solve a **constrained optimization** problem `CEC 2006 <http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC-2006/technical_report.pdf>`_ Test Problem g09), defined as: Find :math:`x^\star = \mathop{\arg\min}\limits_{x}  \,\,\, f(x) \,,` under the constraints :math:`g_i(x) \leq 0.`

Problem Description
---------------------------

We want to solve the problem:

.. math:

   f(x) = (x_1 - 10)^2 + 5(x_2 - 12)^2 + x_3^4  + 3(x_4 - 11)^2 + 10 x_5^6 + 7 x_6^2 + x7^4 - 4 x_6 x_7 - 10  x_6 - 8 x_7 $$

under the four constraints :math:`g_i(x)`:

.. math:

   g_1(x) = -127 + 2x_1^2 + 3 x_2^4 + x_3 + 4x_4^2 + 5x_5
   g_2(x) = -282 + 7x_1 + 3x_2 + 10x_3^2 + x_4 - x_5
   g_3(x) = -196 + 23 x_1 + x_2^2 + 6x_6^2 - 8x_7
   g_4(x) = 4x_1^2 + x_2^2 - 3x_1x_2 + 2x_3^2 + 5x_6 - 11x_7

The Objective Function
---------------------------
Create a folder named `model`. Inside, create a file with name `model.py` and paste the following code,

.. code-block:: python

    #!/usr/bin/env python

    def g09( k ):

      d = k["Parameters"]
      res = (d[0] - 10.0)**2 + 5.0 * (d[1] - 12.0)**2           \
            + d[2]**4  + 3.0 * (d[3] - 11.0)**2                 \
            + 10.0 * d[4]**6 + 7.0 * d[5]**2 + d[6]**4.      \
            - 4.0 * d[5] * d[6] - 10.0 * d[5] - 8.0 * d[6];

      k["Evaluation"] = -res;

This computational model represents our objective function.

For the constraints, add the following code in the same file,

.. code-block:: python

    def g1(k):
      v = k["Parameters"]
      k["Evaluation"] = -127.0 + 2 * v[0] * v[0] + 3.0 * pow(v[1], 4) + v[2] + 4.0 * v[3] * v[3] + 5.0 * v[4]


    def g2(k):
      v = k["Parameters"]
      k["Evaluation"] = -282.0 + 7.0 * v[0] + 3.0 * v[1] + 10.0 * v[2] * v[2] + v[3] - v[4]


    def g3(k):
      v = k["Parameters"]
      k["Evaluation"] = -196.0 + 23.0 * v[0] + v[1] * v[1] + 6.0 * v[5] * v[5] - 8.0 * v[6]

    def g4(k):
      v = k["Parameters"]
      k["Evaluation"] = 4.0 * v[0] * v[0] + v[1] * v[1] - 3.0 * v[0] * v[1] + 2.0 * v[2] * v[2] + 5.0 * v[5] - 11.0 * v[6]

Optimization with (C)CMA-ES
---------------------------
First, open a file and import the korali module

.. code-block:: python

    #!/usr/bin/env python3
    import korali

Import the computational model,

.. code-block:: python

    import sys
    sys.path.append('./model')
    from model import *
    from constraints import *

The Korali Object
---------------------------
Next we construct a `korali.Experiment` object,

.. code-block:: python

    e = korali.Experiment()

Add the objective function and the constraints in the Korali object,

.. code-block:: python

    e["Problem"]["Objective Function"] = g09
    e["Problem"]["Constraints"] = [ g1, g2, g3, g4 ]


The Problem Type
---------------------------
Then, we set the type of the problem to `Direct Evaluation`

.. code-block:: python

    e["Problem"]["Type"] = "Evaluation/Direct/Basic"
    e["Problem"]["Objective"] = "Maximize"


The Variables
---------------------------
We add 7 variables to the experiment and set their domain,

.. code-block:: python

    for i in range(7) :
      e["Variables"][i]["Name"] = "X" + str(i)
      e["Variables"][i]["Lower Bound"] = -10.0
      e["Variables"][i]["Upper Bound"] = +10.0


The Solver
---------------------------
We choose the solver `CMA-ES`,

.. code-block:: python

    e["Solver"]["Type"] = "Optimizer/CMAES"

Then we set a few parameters for CCMA-ES,

.. code-block:: python

    e["Solver"]["Is Sigma Bounded"] = True
    e["Solver"]["Population Size"] = 32
    e["Solver"]["Viability Population Size"] = 4
    e["Solver"]["Termination Criteria"]["Max Value"] = -680.630057374402 - 1e-4
    e["Solver"]["Termination Criteria"]["Max Generations"] = 500

For a detailed description of CCMA-ES settings see :ref:`CMAES <module-solver-optimizer-cmaes>`.

We configure output settings,

.. code-block:: python

    e["File Output"]["Frequency"] = 50
    e["Console Output"]["Frequency"] = 50

Finally, we need to create a Korali `Engine` object add a call to its run() routine, to start the engine.

.. code-block:: python

    k = korali.Engine()
    k.run(e)

Running
---------------------------

We are now ready to run our example:`./run-ccmaes.py`


The results are saved in the folder `_korali_result/`.

Plotting
---------------------------

You can see the results of CMA-ES by running the command,
`python3 -m korali.plotter`
