*****************************************************
TMCMC (Transitional Markov Chain Monte Carlo)
*****************************************************

This is the implementation of the *Transitional Markov Chain Monte Carlo*
algorithm, as published in `Ching2007 <https://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399%282007%29133%3A7%28816%29>`_.

TMCMC avoids sampling from difficult target probability densities
(e.g. posterior distributions in a Bayesian inference problem) but samples from
a series of intermediate PDFs that converge to the target PDF.
This technique is also known as Sampling Importance Resampling in the Bayesian community.
