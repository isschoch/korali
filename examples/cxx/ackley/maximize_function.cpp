#include "korali.h"
#include "models/ackley.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";

 korali["Problem"]["Type"] = "Direct";
 for (int i = 0; i < 4; i++)
 {
  korali["Problem"]["Variables"][i]["Name"] = "X" + std::to_string(i);
  korali["Problem"]["Variables"][i]["Type"] = "Computational";
  korali["Problem"]["Variables"][i]["Distribution"] = "Uniform";
  korali["Problem"]["Variables"][i]["Minimum"] = -32.0;
  korali["Problem"]["Variables"][i]["Maximum"] = +32.0;
 }

 korali["Solver"]["Method"] = "CMA-ES";
 korali["Solver"]["Lambda"] = 10;
 korali["Solver"]["Termination Criteria"]["Max Generations"] = 100;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-12;
 
 korali.run([](Korali::Model::Sequential& d) { m_ackley(d.getParameters(), d.getResults()); });

 return 0;
}
