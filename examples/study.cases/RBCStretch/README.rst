Red Blood Cell Parameter Inference (Stretching) 
=================================================================================
 
Red blooc cells (RBC) are highly deformable objects, allowing for complex dynamical transitions in response to external distur-bances and lay the foundation for understanding the rheology of blood in ﬂows through capillaries, vascular networks and medical devices. The elastic properties of RBCs, for instance, have been the point of interest of many studies due to their direct connection with pathological diseases such as malaria, where RBCs appear to progressively loose their elastic abilities. There is still, however, significant uncertainty in the choice of the mechanical law to describe the RBC properties, as well as in the parameters values of each model. To better understand such ﬂows and simulate whole blood, we have to model the mechanics of a single RBC accurately.

In this experiment, we use a popular approach of modeling the RBC membrane as a collection of particles placed on the nodes of a triangular network. The connectivity is ﬁxed over time, and the particle positions and velocities evolve in time according to Newton’s law. Interactions between neighboring particles are governed by a set of potential functions describing the elastic and viscous mechanical properties of the membrane. Using data for the axial and transverse extensions of RBCs under stretching, we infer the parameters ks and x0, controlling the linear and non-linear elastic components respectively.

We use Korali to run a Bayesian inference sampling experiment using BASIS, a reduced bias variant of the TMCMC algorithm implemented in Korali. The choice of BASIS is supported by the fact that it is one of the most eﬃcient MCMC algorithms in the context of Bayesian uncertainty quantiﬁcation that is targeted to parallel computing architectures. We conﬁgure BASIS to run a population size of 4096 samples per generation. To hold the same statistical assumptions as in the described experiment, we use a likelihood model of type Additive Normal Data. For the computational model for RBC stretching, we used Mirheo, a highperformance GPU-based library for microﬂuidic simulations. We base our choice on the fact that Mirheo has shown exceptionally low time to solution on several benchmark problems, outperforming current state-of-the-art packages. Mirheo is written in C++/CUDA and targets NVIDIA GPUs. 

Scientific Sources
---------------------------

Experiments:

  Mills, J. P., et al. 
  "Nonlinear elastic and viscoelastic deformation of the human red blood cell with optical tweezers." 
  MCB-TECH SCIENCE PRESS- 1 (2004): 169-180.


Numerical work:

  Fedosov, Dmitry A., Bruce Caswell, and George Em Karniadakis. 
  "A multiscale red blood cell model with accurate mechanics, rheology, and dynamics." 
  Biophysical journal 98.10 (2010): 2215-2225.


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

Install python modules
.. code-block:: bash

.. code-block:: bash

	python -m pip install trimesh --user
	python -m pip install mpi4py --user

Running
.. code-block:: bash

.. code-block:: bash

	mir.run --runargs "-n $SLURM_JOB_NUM_NODES --ntasks-per-node=1" ./run.py
