**************************
gridSearch
**************************

This solver implements a grid-search to find the optimum :math:`x^\star` for a given set of parameters :math:`x=(x_1,\dots,x_n)`. Therefore it computes the value for the given objective function

.. math::

   f(x_i)\quad \forall i=1,\dots, N

and returns the optimum over the so computed values

.. math::

   f(x^\star) = \max \{f(x_1),\dots,f(x_N)\}

and the corresponding argument of the maximum :math:`x^\star`.
