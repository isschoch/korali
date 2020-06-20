#include "assert.h"
#include "korali.hpp"

void minmodel1(korali::Sample &s)
{
  double x = s["Parameters"][0];
  double eval = (x - 2.0) * (x - 2.0) + 10;
  s["Evaluation"] = -eval;
}

int main(int argc, char **argv)
{
  auto k = korali::Engine();
  auto e = korali::Experiment();

  e["Problem"]["Type"] = "Optimization";
  e["Problem"]["Objective Function"] = &minmodel1;

  e["Variables"][0]["Name"] = "X";
  e["Variables"][0]["Lower Bound"] = -10.0;
  e["Variables"][0]["Upper Bound"] = +10.0;

  e["Solver"]["Type"] = "Optimizer/CMAES";
  e["Solver"]["Population Size"] = 32;
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-8;
  e["Solver"]["Termination Criteria"]["Max Generations"] = 100;

  k.run(e);

  std::vector<double> xopt = e["Solver"]["Best Ever Variables"];
  double fopt = e["Solver"]["Best Ever Value"];

  assert(std::abs((fopt - -10.0) / fopt) <= 1e-6);
  assert(std::abs((xopt[0] - 2.0) / fopt) <= 1e-3);

  return 0;
}

/*

x = e["Solver"]["Current Best Variables"]
f = e["Solver"]["Current Best Value"]

assertclose(2.0, xopt, 1e-3)
assertclose(2.0, x, 1e-3)
assertclose(10.0, fopt, 1e-6)
assertclose(10.0, f, 1e-4)
*/
