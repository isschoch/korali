**************************
Optimization
**************************

Solves the optimization problem of continuous/discrete variables, given a model function :math:`f(x)` given the form:

.. math::

   \begin{align}
   &\underset{x}{\operatorname{maximize}}& & f(x) \\
   &\operatorname{subject\;to}
   & &g_i(x) \leq 0, \quad i = 1,\dots,m
   \end{align}
   

Where:

  + :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}`: is the *objective function* to be maximized over the :math:`n`-variable vector :math:`x`
  
  + :math:`g_i(x) \leq 0`: are a set of inequality contraints to be satisfied.
 