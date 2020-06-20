Running MPI Applications
=====================================================

In this tutorial we show how an MPI model can be executed with Korali.

MPI Init
---------------------------

Do not forget to init MPI inside the Korali application:

.. code-block:: python

    MPI_Init(&argc, &argv);

Distributed Conduit
---------------------------

Run with the `Distributed` conduit to benefit from parallelized model evaluations.
Note that we set `Workers Per Team` to assign a team of MPI processes to the model.

.. code-block:: python

    k["Conduit"]["Type"] = "Distributed";
    k["Conduit"]["Workers Per Team"] = workersPerTeam;
    k["Profiling"]["Detail"] = "Full";
    k["Profiling"]["Frequency"] = 0.5;

Run
---------------------------

Compile the script with the `Makefile`.
Run the script with an input argument (Workers Per Team):

.. code-block:: bash

    ./run-tmcmc 4
