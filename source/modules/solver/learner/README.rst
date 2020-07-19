************
Deep Learner
************

Uses a :ref:`Neural Network <module-neuralnetwork>` to solve a :ref:`Supervised Learning <module-problem-supervisedlearning>` problem.

It employs three Neural Networks:

- *Training Network* - Used to adjust the weights and biases with the help of a user-defined optimizer to minimize the loss function given in the supervised learning problem (e.g., Mean Square Error).

- *Validation Network* - Used to measure the improvement (in terms of loss function values) of the training network on the validation data given in the supervised learning problem.
  
- *Test Network* - The result of the optimisation procedure which can be used to evaluate the neural network on a test set using the test() function.

