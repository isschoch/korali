#include "_model/jacobi.h"
#include "korali.hpp"
#include <unistd.h>

int main(int argc, char *argv[])
{
  int workersPerTeam = 1;

  if (argc == 2)
  {
    workersPerTeam = atoi(argv[1]);
    if (64 % workersPerTeam != 0)
    {
      fprintf(stderr, "Command Line Argument (Workers Per Team) must be divisor of 64! exit..)\n");
      exit(-1);
    }
  }

  MPI_Init(&argc, &argv);

  auto e = korali::Experiment();

  e["Problem"]["Type"] = "Bayesian/Reference";
  e["Problem"]["Likelihood Model"] = "Normal";
  e["Problem"]["Reference Data"] = getReferenceData();
  e["Problem"]["Computational Model"] = &jacobi;

  e["Distributions"][0]["Name"] = "Uniform 0";
  e["Distributions"][0]["Type"] = "Univariate/Uniform";
  e["Distributions"][0]["Minimum"] = +0.0;
  e["Distributions"][0]["Maximum"] = +5.0;

  e["Distributions"][1]["Name"] = "Uniform 1";
  e["Distributions"][1]["Type"] = "Univariate/Uniform";
  e["Distributions"][1]["Minimum"] = -5.0;
  e["Distributions"][1]["Maximum"] = +0.0;

  e["Distributions"][2]["Name"] = "Uniform 2";
  e["Distributions"][2]["Type"] = "Univariate/Uniform";
  e["Distributions"][2]["Minimum"] = +1.0;
  e["Distributions"][2]["Maximum"] = +6.0;

  e["Distributions"][3]["Name"] = "Uniform 3";
  e["Distributions"][3]["Type"] = "Univariate/Uniform";
  e["Distributions"][3]["Minimum"] = -4.0;
  e["Distributions"][3]["Maximum"] = +1.0;

  e["Distributions"][4]["Name"] = "Uniform 4";
  e["Distributions"][4]["Type"] = "Univariate/Uniform";
  e["Distributions"][4]["Minimum"] = +2.0;
  e["Distributions"][4]["Maximum"] = +7.0;

  e["Distributions"][5]["Name"] = "Uniform 5";
  e["Distributions"][5]["Type"] = "Univariate/Uniform";
  e["Distributions"][5]["Minimum"] = -3.0;
  e["Distributions"][5]["Maximum"] = +2.0;

  e["Distributions"][6]["Name"] = "Uniform 6";
  e["Distributions"][6]["Type"] = "Univariate/Uniform";
  e["Distributions"][6]["Minimum"] = 0.0;
  e["Distributions"][6]["Maximum"] = +20.0;

  e["Variables"][0]["Name"] = "X0";
  e["Variables"][0]["Prior Distribution"] = "Uniform 0";
  e["Variables"][0]["Initial Value"] = +2.0;
  e["Variables"][0]["Initial Standard Deviation"] = +1.0;

  e["Variables"][1]["Name"] = "X1";
  e["Variables"][1]["Prior Distribution"] = "Uniform 1";
  e["Variables"][1]["Initial Value"] = -2.0;
  e["Variables"][1]["Initial Standard Deviation"] = +1.0;

  e["Variables"][2]["Name"] = "Y0";
  e["Variables"][2]["Prior Distribution"] = "Uniform 2";
  e["Variables"][2]["Initial Value"] = +4.0;
  e["Variables"][2]["Initial Standard Deviation"] = +1.0;

  e["Variables"][3]["Name"] = "Y1";
  e["Variables"][3]["Prior Distribution"] = "Uniform 3";
  e["Variables"][3]["Initial Value"] = -2.0;
  e["Variables"][3]["Initial Standard Deviation"] = +1.0;

  e["Variables"][4]["Name"] = "Z0";
  e["Variables"][4]["Prior Distribution"] = "Uniform 4";
  e["Variables"][4]["Initial Value"] = +5.0;
  e["Variables"][4]["Initial Standard Deviation"] = +1.0;

  e["Variables"][5]["Name"] = "Z1";
  e["Variables"][5]["Prior Distribution"] = "Uniform 5";
  e["Variables"][5]["Initial Value"] = +0.0;
  e["Variables"][5]["Initial Standard Deviation"] = +1.0;

  e["Variables"][6]["Name"] = "[Sigma]";
  e["Variables"][6]["Prior Distribution"] = "Uniform 6";
  e["Variables"][6]["Initial Value"] = +10.0;
  e["Variables"][6]["Initial Standard Deviation"] = +1.0;

  e["Solver"]["Type"] = "Optimizer/CMAES";
  e["Solver"]["Population Size"] = 16;
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7;
  e["Solver"]["Termination Criteria"]["Max Generations"] = 10;

  auto k = korali::Engine();

  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Workers Per Team"] = workersPerTeam;
  k["Conduit"]["Communicator"] = MPI_COMM_WORLD;
  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Frequency"] = 0.5;

  k.run(e);

  MPI_Finalize();
  return 0;
}
