******************************************************************
CMAES (Covariance Matrix Adaptation Evolution Strategy)
******************************************************************

This is the implementation of the *Covariance Matrix Adaptation Evolution Strategy*, as published in `Hansen2006 <https://doi.org/10.1007/3-540-32494-1_4>`_.
In an evolution strategy, new candidate solutions are sampled according to a multivariate normal distribution in :math:`\mathbb {R} ^{n}`. Recombination amounts to selecting a new mean value for the distribution. Mutation amounts to adding a random vector, a perturbation with zero mean. Pairwise dependencies between the variables in the distribution are represented by a covariance matrix. The covariance matrix adaptation (CMA) is a method to update the covariance matrix of this distribution.
CMAES works iteratively, evaluating a number :math:`\lambda` of samples per generation, and improving the covariance matrix for the samples in the next generation.

This solver also implements the *Constrained Covariance Matrix Adaptation Evolution Strategy*, as published in `Arampatzis2019 <https://dl.acm.org/citation.cfm?doid=3324989.3325725>`_.

In an evolution strategy, new candidate solutions are sampled according to a multivariate normal distribution in :math:`\mathbb {R} ^{n}`. Recombination amounts to selecting a new mean value for the distribution. Mutation amounts to adding a random vector, a perturbation with zero mean. Pairwise dependencies between the variables in the distribution are represented by a covariance matrix. The covariance matrix adaptation (CMA) is a method to update the covariance matrix of this distribution.
CCMAES is an extension of CMAES for constrained optimization problems. It uses the principle of *viability boundaries* to find an initial mean vector for the proposal distribution that does not violate constraints, and secondly it uses a  *constraint handling technique* to efficiently adapt the proposal distribution to the constraints.


