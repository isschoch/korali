/* In this example, we demonstrate how Korali finds values for the
variables that maximize the objective function, given by a
user-provided computational model. */

#include "_model/direct.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  // Configuring Problem.
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Optimization";
  e["Problem"]["Objective Function"] = &direct;

  // Defining the problem's variables.
  e["Variables"][0]["Name"] = "X";
  e["Variables"][0]["Lower Bound"] = -10.0;
  e["Variables"][0]["Upper Bound"] = +10.0;

  // Configuring CMA-ES parameters
  e["Solver"]["Type"] = "Optimizer/CMAES";
  e["Solver"]["Population Size"] = 32;
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7;
  e["Solver"]["Termination Criteria"]["Max Generations"] = 100;

  // Running Korali
  auto k = korali::Engine();
  k.run(e);
}
