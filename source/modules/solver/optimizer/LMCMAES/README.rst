****************************************************************************
LM-CMAES (Limited Memory - Covariance Matrix Adaptation Evolution Strategy)
****************************************************************************

This is the implementation of the *Limited Memory - Covariance Matrix Adaptation Evolution Strategy*, as published in `Loshchilov2015 <http://loshchilov.com/publications/LMCMA.pdf>`_.

For a general overview on evolution strategies, please refer to: :ref:`CMAES <module-solver-optimizer-cmaes>`.

In LM-CMAES the memory footprint of the decomposition of the covariance matrix is reduced. LM-CMAES is said to outperform the original CMA-ES on non-separable ill-conditioned problems with a factor increasing with problem dimension (see reference). CMAES works iteratively, evaluating a number :math:`\lambda` of samples per generation, and improving the covariance matrix for the samples in the next generation.

