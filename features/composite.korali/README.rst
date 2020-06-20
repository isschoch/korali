Korali + Korali Model
==============================================

This tutorial demonstrates that Korali can be composed: models evaluations can contain in themselves a new Korali experiment. 

In this example, the main experiment describes an optimization over the parameter X. Each evaluation will optimize over the parameter Y to find the value of Y which maximizes X*Y.

Both, the main Korali and subsequent Korali instances run concurrently using 2 workers each, resulting in a 2 + 2x2 = 6 Korali workers instantiated.
