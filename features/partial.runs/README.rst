Partial Runs
=====================================================

In this tutorial we show how to stop and restart a previous Korali run.
During the first and the second phase you may adjust a solvers settings.
All scripts in this tutorial follow this structure:

Run
---------------------------

Set a `Termination Criteria` and run:

.. code-block:: python

    print('------------------------------------------------------')
    print('Now running first 50 generations...')
    print('------------------------------------------------------')

    e["Solver"]["Termination Criteria"]["Max Generations"] = 50
    k.run(e)

Restart
---------------------------

Update `Termination Criteria` and restart with `run`:

.. code-block:: python

    print('------------------------------------------------------')
    print('Now running last 50 generations...')
    print('------------------------------------------------------')

    e["Solver"]["Termination Criteria"]["Max Generations"] = 100
    k.run(e)

