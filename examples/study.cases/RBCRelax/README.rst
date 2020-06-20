Red Blood Cell Parameter Inference (Relaxation)
================================================================================= 

An interesting aspect of RBCs that have been observed experimentally is the eﬀect that viscosity has on relaxation time of an RBC membrane. This phenomenon is, however, still not fully understood from a computational point of view. To gain a deeper understanding of the viscosity pa-rameter on RBC computational models, we studied the pairwise dissipation interaction between neighboring ver-tices on the triangular mesh. One of the unambiguous experiments to determine the membrane viscosity is the relaxation of a stretched RBC to its equilibrium shape. Due to the presence of heterogeneous data from ﬁve dif-ferent experimental studies, we formulated the inference problem for the parameter governing the dissipation in-teraction as a two-staged hierarchical Bayesian inference to estimate the membrane dissipation and its uncertainty, as well as reason about the validity of the RBC model implemented in Mirheo.

In the ﬁrst stage, we inferred ﬁve posterior distributions for the dissipation parameter, each conditioned on an individual data set. To infer these distributions, we deﬁned a likelihood function comprising the execution of a virtual RBC relaxation experiment in Mirheo that allows us to compare simulated length scales to experimental measurements. In the second stage, we assumed that all the posterior distributions found in the ﬁrst stage follow a generalized distribution that is controlled by some hyperparameters which remained to be inferred.

We performed the sampling in stage one and two using BASIS implemented in Korali, whereas we used 512 samples in stage one, and 10000 samples in stage two. Finally we found that the estimated membrane dissipation parameter, by the means of the maximum a posteriori (MAP), corresponds to membrane viscosity parameters estimated in literature.

In this case study, we sample ﬁve posterior distribu-tions of the membrane dissipation parameter (γC) during stage one. All ﬁve experiments share a similar setup and diﬀer only in the initial conditions and the reference data. 

Scientific Sources
---------------------------

Data in set 1 in [data/hochmuth_1979](data/hochmuth_1979):

  Hochmuth, R. M., Worthy, P. R., & Evans, E. A. (1979). 
  Red cell extensional recovery and the determination of membrane viscosity. 
  Biophysical Journal, 26(1), 101–114. 
  https://doi.org/10.1016/S0006-3495(79)85238-8

Data set 2 in [data/henon_1999](data/henon_1999):

  Hénon, S., Lenormand, G., Richert, A., & Gallet, F. (1999). 
  A new determination of the shear modulus of the human erythrocyte membrane using optical tweezers. 
  Biophysical Journal, 76(2), 1145–1151. 
  https://doi.org/10.1016/S0006-3495(99)77279-6

This experiment is based on [this](https://github.com/cselab/mirRun/tree/master/rbc_relaxation/) setup. 

Results Report
---------------------------------------------------

D. Wälchli, S. Martin, A. Economides, L. Amoudruz, G. Arampatzis, P. Koumoutsakos, Load Balancing in Large Scale Bayesian Inference, in: Proceedings of the The Platform for Advanced Scientiﬁc Computing (PASC) Conference 2020, to appear, 2020


Setup
----------------------------------------------------------------------

**Installation on Piz Daint**

Install Mirheo Tools

.. code-block:: bash

  cd $HOME
  mkdir src
  cd src
  git clone --recursive https://github.com/cselab/Mirheo.git
  cd Mirheo
  cd tools
  ./configure --bin-prefix $HOME/.local/bin --exec-cmd srun
  make 
  make install

Install Mirheo 

.. code-block:: bash

  cd $HOME/src/Mirheo
  . mir.load
  mir.make install
  cd ..
  rm -rf Mirheo

Install python modulddes

.. code-block:: bash

  python -m pip install trimesh --user
  python -m pip install mpi4py --user

Running

.. code-block:: bash

  mir.run --runargs "-n $SLURM_JOB_NUM_NODES --ntasks-per-node=1" ./run.py
