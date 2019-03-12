#include "korali.h"
#include "model/gaussian.h"

int main(int argc, char* argv[])
{
	gaussian_init();

	auto problem = Korali::DirectEvaluation(gaussian);

  Korali::Parameter p;
  p.setBounds(-32.0, +32.0);
	for (int i = 0; i < NDIMS; i++) problem.addParameter(p);

  auto Solver = Korali::KoraliCMAES(&problem);
	Solver.setStopMinDeltaX(1e-11);
	Solver.setLambda(128);
	Solver.run();

	return 0;
}