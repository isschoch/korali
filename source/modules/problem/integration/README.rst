**************************
Integration
**************************

The integration problem is considering the numerical approximation of integrals

.. math::

  I = \int f(x)\mathrm{d}x

using 

.. math::

  I \approx \sum\limits_{i=1}^N w_i f(x_i)

The supported methods are Monte Carlo Integration and Quadrature. For Quadrature the weights for the Rectangle rule, the Trapezoidal rule and the Simpson rule are given, and there is the possibility to provide own weights and evaluation points.

