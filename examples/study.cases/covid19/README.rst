*************************************************
SIR Epidemiology Model
*************************************************

We use the Bayesian paradigm in order to find the parameters of the `SIR model <https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model_without_vital_dynamics>`_.

The uncertainty in the parameters is quantified by sampling the conditional distribution of the parameters on available data using `Korali <https://www.cse-lab.ethz.ch/korali/>`_.

We use data from infected people on coronavirus in Switzerland. The data are obtained from `here <https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv>`_.

The data at time,  :math:`d_t` correspond to the model quantity

.. math::

  N - S(t)

where  :math:`S(t)` is the variable that models the number of the healthy population in the SIR model and `N` is the total size of the population.

The uncertainty in the parameters is propagated in the model and predictions for the future can be made.

To run the code, first `install korali <https://www.cse-lab.ethz.ch/korali/docs/using/install.html>`_.

Then, download this repository

.. code-block:: bash

   git clone https://github.com/cselab/korali.git
   cd examples/study.cases/covid

Make sure you have installed the latest version (at least 1.4) of the Scipy library:

.. code-block:: bash

   python3 -m pip install scipy==1.4.1

Also, get the pybind11 development source code by running:

.. code-block:: bash

   ./install_prereqs.sh

Run the code main code,

.. code-block:: bash

  ./main.py --country greece           \
            --populationSize 10000000  \
            --nSamples 2000            \
            --nThreads 4               \
            --futureDays 2             \
            --validateData 1           \
            --percentages 0.5 0.99



Check the sampled distribution with

.. code-block:: bash

   python3 -m korali.plotter --dir data/_korali_samples/

In the diagonal of the figure, the marginal distribution of the parameters is shown in order, beta, gamma and sigma. The off-diagonal elements of the figure are plots of the pair the marginal distributions.



.. [Commenting this out until the image is back in again] .. image:: data/figures/prediction.png


C++-optimized version
=====================

By default, the Python scripts will use scipy for integration. To use a C++ version, perform the following steps starting from the ``5.coronavirus`` folder:

.. code-block:: bash

    git submodule update --init --recursive
    mkdir -p build
    cd build
    cmake ..
    make -j4

The ``sample.py`` script will automatically detect if the C++ version is available.
