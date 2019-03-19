#ifdef KORALI_UPCXX_CONDUIT
#include "conduits/upcxx.h"
#include "solvers/base.h"

Korali::Conduit::Conduit(BaseSolver* solver) : Korali::BaseConduit::BaseConduit(solver)
{
	_rankId = 0;
	_rankCount = 1;
  _continueEvaluations = true;
	_evaluateSample = false;
}

void Korali::Conduit::initialize()
{
	_k = this;

	upcxx::init();
	_rankId = upcxx::rank_me();
	_rankCount = upcxx::rank_n();

	if (_rankCount < 2) { fprintf(stderr, "[Korali] Error: Running Korali's UPCxx Conduit with less than 2 ranks is not allowed.\n"); exit(-1); }

  // Allocating Global Pointer for Samples
	if (_rankId == 0) sampleGlobalPtr  = upcxx::new_array<double>(_solver->N*_solver->_sampleCount);
	upcxx::broadcast(&sampleGlobalPtr,  1, 0).wait();

  if (_rankId == 0) supervisorThread(); else workerThread();

  upcxx::barrier();
  upcxx::finalize();
}

void Korali::Conduit::supervisorThread()
{
  // Creating Worker Queue
  for (int i = 1; i < _rankCount; i++) _workers.push(i);

	_solver->runSolver();

  for (int i = 1; i < _rankCount; i++) upcxx::rpc_ff(i, [](){_k->_continueEvaluations = false;});
}

double* Korali::Conduit::getSampleArrayPointer()
{
  return sampleGlobalPtr.local();
}

void Korali::Conduit::workerThread()
{
	while(_continueEvaluations)
	{
		if (_evaluateSample)
		{
			_evaluateSample = false;
			double candidatePoint[_solver->N];
			upcxx::rget(sampleGlobalPtr + _sampleId*_solver->N, candidatePoint, _solver->N).wait();
			double candidateFitness = _solver->_problem->evaluateFitness(candidatePoint);
			//printf("Worker %d: Evaluated %ld:[%f, %f] - Fitness: %f\n", _rankId, _sampleId, candidatePoint[0], candidatePoint[1], candidateFitness);
			upcxx::rpc_ff(0, [](size_t c, double fitness, int workerId){_k->_solver->processSample(c, fitness); _k->_workers.push(workerId); }, _sampleId, candidateFitness, _rankId);
		}
		 upcxx::progress();
	}
}

void Korali::Conduit::evaluateSample(size_t sampleId)
{
	while(_workers.empty()) upcxx::progress();
	int workerId = _workers.front(); _workers.pop();
	upcxx::rpc_ff(workerId, [](size_t c){_k->_sampleId = c; _k->_evaluateSample = true;}, sampleId);
}

void Korali::Conduit::checkProgress()
{
	upcxx::progress();
}
#endif
