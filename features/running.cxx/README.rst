Running C++ 
=====================================================

In this tutorial we show how Korali can be used with c++.
For this we optimize a model with the solver `CMA-ES` and `LM-CMA`. Here we want to find the parameters :math:`v = (Intensity , PosX, PosY, Sigma)` that maximize the posterior in a Bayesian problem.  

How to run the example
---------------------------

Run the `Makefile` to compile the executables. Then you can run am example, e.g. `./run-cmaes`
This should output information about the process and result of the optimization.


Short explanation
---------------------------

The problem to be solved is a static heat conduction problem, with
a candle as static heat source. The variables `Intensity` , `PosX`, `PosY` are position and intensity of the candle. `Sigma` is the standard deviation of the noise in the
`Additive Normal` noise model - the noise :math:`\epsilon` that is added to the function :math:`f` (heat2Dsolver, see below) to obtain the measured temperature at each data point.


Computational Model and Data Points
---------------------------------------

First, we create the Korali engine and an experiment that we will configure,

.. code-block:: python

    auto k = korali::Engine();
    auto e = korali::Experiment();
    auto p = heat2DInit(&argc, &argv);


Here, `heat2DInit`, defined in [heat2d.cpp](model/heat2d.cpp), returns the data points (triples (xPos, yPos, refTemp)) as `p`. We model refTemp as a function of xPos and yPos (a function whose parameters :math:`v1` we want to determine), in addition to some noise: :math:`refTemp(xPos, yPos) = f_{v1}(xPos, yPos) + \epsilon`. The distribution of the noise :math:`\epsilon` depends on parameters :math:`v2`. We want to estimate :math:`v = (v1, v2)`.  

We next set the problem type to Bayesian inference, assign the objective values (refTemp values) of our data as `Reference Data` and set the computational model to the function `heat2DSolver` (our `f` above), defined in [heat2d.cpp](model/heat2d.cpp),

.. code-block:: python

    e["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Reference";
    e["Problem"]["Likelihood Model"] = "Additive Normal";
    e["Problem"]["Reference Data"] = p.refTemp;
    e["Problem"]["Computational Model"] = &heat2DSolver;

Function [`heat2DSolver`](model/heat2d.cpp) internally already has access to the data points created by `heat2DInit`. The function calculates temperature values iteratively on a grid over the domain of xPos and yPos, using the Gauss-Seidel method. To get the temperature values at the data points and set them as `Reference Evaluations`, `heat2DSolver` finds a point on the grid close to each data point and returns the temperature value at this grid point.

Solver
---------------------------

Then, we decide on `CMAES` as solver and configure its parameters,

.. code-block:: python

    e["Solver"]["Type"] = "Optimizer/CMAES";
    e["Solver"]["Population Size"] = 32;
    e["Solver"]["Termination Criteria"]["Max Generations"] = 100;

Variables and Prior Distributions
------------------------------------

We then need to define four variables, as well as a prior distribution for each
of them,

.. code-block:: python

    e["Distributions"][0]["Name"] = "Uniform 0";
    e["Distributions"][0]["Type"] = "Univariate/Uniform";
    e["Distributions"][0]["Minimum"] = 10.0;
    e["Distributions"][0]["Maximum"] = 60.0;

    e["Distributions"][1]["Name"] = "Uniform 1";
    e["Distributions"][1]["Type"] = "Univariate/Uniform";
    e["Distributions"][1]["Minimum"] = 0.0;
    e["Distributions"][1]["Maximum"] = 0.5;

    e["Distributions"][2]["Name"] = "Uniform 2";
    e["Distributions"][2]["Type"] = "Univariate/Uniform";
    e["Distributions"][2]["Minimum"] = 0.6;
    e["Distributions"][2]["Maximum"] = 1.0;

    e["Distributions"][3]["Name"] = "Uniform 3";
    e["Distributions"][3]["Type"] = "Univariate/Uniform";
    e["Distributions"][3]["Minimum"] = 0.0;
    e["Distributions"][3]["Maximum"] = 20.0;

    e["Variables"][0]["Name"] = "Intensity";
    e["Variables"][0]["Bayesian Type"] = "Computational";
    e["Variables"][0]["Prior Distribution"] = "Uniform 0";
    e["Variables"][0]["Initial Mean"] = 30.0;
    e["Variables"][0]["Initial Standard Deviation"] = 5.0;
    
    e["Variables"][1]["Name"] = "PosX";
    e["Variables"][1]["Bayesian Type"] = "Computational";
    e["Variables"][1]["Prior Distribution"] = "Uniform 1";
    e["Variables"][1]["Initial Mean"] = 0.25;
    e["Variables"][1]["Initial Standard Deviation"] = 0.01;

    e["Variables"][2]["Name"] = "PosY";
    e["Variables"][2]["Bayesian Type"] = "Computational";
    e["Variables"][2]["Prior Distribution"] = "Uniform 2";
    e["Variables"][2]["Initial Mean"] = 0.8;
    e["Variables"][2]["Initial Standard Deviation"] = 0.1;

    e["Variables"][3]["Name"] = "Sigma";
    e["Variables"][3]["Bayesian Type"] = "Statistical";
    e["Variables"][3]["Prior Distribution"] = "Uniform 3";
    e["Variables"][3]["Initial Mean"] = 10.0;
    e["Variables"][3]["Initial Standard Deviation"] = 1.0;

Running the Optimization
---------------------------
Finally, we call the `run()` routine to run the optimization, to find those
parameters v that are most likely, using Bayes rule: We want to find v that
maximize :math:`P(v|X) = P(X|v)*prior(v)`, i.e, the likelihood of
the data times their prior.

.. code-block:: python

    k.run(e);
